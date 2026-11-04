"""Optional HTTP transport for sequential CAD agents. Run with uvicorn factory."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .backend import FreeCADBackend
from .environment import CADAgentEnv
from .observations import agent_messages, policy_images, transition
from .tasks import make_task


class SessionRequest(BaseModel):
    family: Literal["plate", "bushing", "flange", "stepped_shaft"] = "plate"
    seed: int = 0
    mode: Literal["create", "edit", "repair"] = "create"


class DatasetSessionRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=128)


class ActionRequest(BaseModel):
    action: dict[str, Any]


def create_app(
    work_dir="artifacts/api-sessions",
    executable=None,
    *,
    visual=None,
    dataset_dir=None,
    train_only=None,
):
    app = FastAPI(title="Verified FreeCAD Agent Environment", version="0.2.0")
    sessions = {}
    registry_lock = Lock()
    backend = FreeCADBackend(executable)
    dataset_dir = dataset_dir or os.environ.get("FREECAD_DATASET")
    visual = (
        os.environ.get("FREECAD_VISUAL", "1" if dataset_dir else "0") == "1"
        if visual is None
        else visual
    )
    train_only = (
        os.environ.get("FREECAD_TRAIN_ONLY", "0") == "1"
        if train_only is None
        else train_only
    )
    dataset = Path(dataset_dir).resolve() if dataset_dir else None
    accepted_tasks = set()
    if dataset:
        from cad_env.learning.audit import audit_dataset

        audit = audit_dataset(dataset)
        summary = json.loads((dataset / "summary.json").read_text())
        if summary.get("benchmark", "").startswith("visual-grounding-") or summary.get(
            "families"
        ) == ["visual_grounding"]:
            from cad_env.learning.visual_audit import audit_visual

            audit = audit_visual(dataset, base_report=audit)
        elif summary.get("benchmark", "").startswith("complex-cad-"):
            from cad_env.learning.curriculum_metrics import audit_complex

            audit = audit_complex(dataset, base_report=audit)
        if not audit["ok"]:
            raise ValueError("Configured dataset failed its audit")
        accepted_tasks = {
            e["task_id"]
            for line in (dataset / "manifest.jsonl").read_text().splitlines()
            if (e := json.loads(line))["accepted"]
        }

    def public_observation(observation, sid, assets):
        public = dict(observation)
        for key in ("goal_images", "images"):
            if key not in observation:
                continue
            public[key] = []
            for image in observation[key]:
                path = Path(image).resolve()
                token = hashlib.sha256(path.read_bytes()).hexdigest()
                assets[token] = path
                public[key].append(f"/sessions/{sid}/images/{token}")
        return public

    def new_session(task):
        if train_only and task["split"] != "train":
            raise HTTPException(403, "Training sessions cannot consume held-out tasks")
        if task.get("goal_images") and not visual:
            raise HTTPException(409, "Visual-goal tasks require FREECAD_VISUAL=1")
        env = CADAgentEnv(Path(work_dir), backend, visual=visual, asset_root=dataset)
        observation, _ = env.reset(task)
        sid = uuid.uuid4().hex
        assets = {}
        public = public_observation(observation, sid, assets)
        with registry_lock:
            sessions[sid] = (env, Lock(), assets, [])
        return {
            "session_id": sid,
            "observation": public,
            "messages": agent_messages(task, observation, visual=visual),
            "images": policy_images(public),
        }

    # Research transport executes native work sequentially within this process.
    # Use factory worker pools for parallel workloads.
    @app.get("/health")
    async def health():
        return {
            "backend": "freecad",
            "executable": backend.executable,
            "simulation": False,
            "visual": visual,
            "dataset_loaded": dataset is not None,
            "dataset_train_only": train_only,
        }

    @app.post("/sessions")
    async def create(request: SessionRequest):
        return new_session(make_task(request.family, request.seed, request.mode))

    @app.post("/dataset/sessions")
    async def create_dataset_session(request: DatasetSessionRequest):
        if dataset is None:
            raise HTTPException(503, "Configure FREECAD_DATASET for dataset sessions")
        if request.task_id not in accepted_tasks:
            raise HTTPException(404, "Unknown accepted task")
        from cad_env.learning.task_identity import load_private_task

        try:
            task = load_private_task(
                dataset / "episodes", request.task_id, require_train=train_only
            )
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc
        return new_session(task)

    @app.get("/sessions/{session_id}/images/{asset_id}")
    async def image(session_id: str, asset_id: str):
        with registry_lock:
            entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "Unknown session")
        with entry[1]:
            path = entry[2].get(asset_id)
            if path is None:
                raise HTTPException(404, "Unknown public image")
        return Response(content=path.read_bytes(), media_type="image/png")

    @app.post("/sessions/{session_id}/step")
    async def step(session_id: str, request: ActionRequest):
        with registry_lock:
            entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "Unknown session")
        env, lock, assets, history = entry
        with lock:
            try:
                observation, reward, terminated, truncated, info = env.step(
                    request.action
                )
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            info = {k: v for k, v in info.items() if k != "result_dir"}
            history.append(
                {
                    **transition(request.action, info, observation),
                    "response": json.dumps(request.action)[:2000],
                }
            )
            public = public_observation(observation, session_id, assets)
            return {
                "observation": public,
                "messages": agent_messages(
                    env.task, observation, history, visual=visual
                ),
                "images": policy_images(public),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }

    @app.delete("/sessions/{session_id}")
    async def close(session_id: str):
        with registry_lock:
            entry = sessions.pop(session_id, None)
        if entry is None:
            raise HTTPException(404, "Unknown session")
        with entry[1]:
            entry[0].closed = True
        return {"closed": True}

    return app
