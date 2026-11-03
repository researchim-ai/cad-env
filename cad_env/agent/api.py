"""Optional HTTP transport for sequential CAD agents. Run with uvicorn factory."""

import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .backend import FreeCADBackend
from .environment import CADAgentEnv
from .tasks import make_task


class SessionRequest(BaseModel):
    family: Literal["plate", "bushing", "flange", "stepped_shaft"] = "plate"
    seed: int = 0
    mode: Literal["create", "edit", "repair"] = "create"


class ActionRequest(BaseModel):
    action: dict[str, Any]


def create_app(work_dir="artifacts/api-sessions", executable=None):
    app = FastAPI(title="Verified FreeCAD Agent Environment", version="0.2.0")
    sessions = {}
    registry_lock = Lock()
    backend = FreeCADBackend(executable)

    # Research transport executes native work sequentially within this process.
    # Use factory worker pools for parallel workloads.
    @app.get("/health")
    async def health():
        return {
            "backend": "freecad",
            "executable": backend.executable,
            "simulation": False,
        }

    @app.post("/sessions")
    async def create(request: SessionRequest):
        env = CADAgentEnv(Path(work_dir), backend)
        observation, _ = env.reset(
            make_task(request.family, request.seed, request.mode)
        )
        sid = uuid.uuid4().hex
        with registry_lock:
            sessions[sid] = (env, Lock())
        return {"session_id": sid, "observation": observation}

    @app.post("/sessions/{session_id}/step")
    async def step(session_id: str, request: ActionRequest):
        with registry_lock:
            entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "Unknown session")
        env, lock = entry
        with lock:
            try:
                observation, reward, terminated, truncated, info = env.step(
                    request.action
                )
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            info = {k: v for k, v in info.items() if k != "result_dir"}
            return {
                "observation": observation,
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
