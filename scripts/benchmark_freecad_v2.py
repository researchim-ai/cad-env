"""Opt-in single-action VLM benchmark for the expanded FreeCAD contract.

Uses development-validation only. Does not alter or pool frozen v1 results.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from cad_env.drawings.batch import write_json
from cad_env.freecad_v2.backend import Backend
from cad_env.freecad_v2.environment import Environment, messages
from cad_env.learning.audit import audit_dataset
from cad_env.learning.policy import HFPolicy


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--profile", choices=["renders", "iso", "gost"], default="renders")
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU 1 only")
    if args.limit < 1:
        p.error("Positive limit required")
    root = args.dataset.absolute()
    audit = audit_dataset(root)
    if not audit["ok"]:
        raise ValueError(audit["errors"])
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("cad_contract") != "freecad.v2":
        raise ValueError("Expanded FreeCAD dataset required")
    if summary.get("drawing_profiles") and args.profile == "renders":
        raise ValueError(
            "Choose --profile iso or --profile gost for the drawing dataset"
        )
    cases = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    cases = sorted(
        (c for c in cases if c["accepted"] and c["split"] == "validation"),
        key=lambda c: c["task_id"],
    )[: args.limit]
    if not cases:
        raise ValueError("No development-validation cases")
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend()
    status = {
        "status": "loading",
        "contract": "freecad.v2",
        "selected": [c["task_id"] for c in cases],
        "protocol": {
            "profile": args.profile,
            "max_actions": 128,
            "greedy": True,
            "goal_views": ["iso", "top", "front"]
            if args.profile == "renders"
            else "native views and dimensional appendix crops",
            "input": "dimensioned text + goal renders + current public JSON"
            if args.profile == "renders"
            else "goal drawing crops + current public JSON; no separate specification transcription",
            "max_context": 32768,
        },
        "episodes": [],
        "dataset_summary_sha256": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
        "source_hashes": backend.sources,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Development validation on expanded feature probes; separate from ISO/GOST v1 benchmark.",
    }
    path = args.output / "status.json"
    write_json(path, status)
    try:
        policy = HFPolicy(
            args.model,
            max_context=32768,
            max_pixels=327680,
            temperature=0,
            max_new_tokens=1024,
            chat_template_kwargs={"enable_thinking": False},
        )
        status.update(status="evaluating", model_identity=policy.identity)
        write_json(path, status)
        for case in cases:
            task = json.loads(
                (root / "episodes" / case["task_id"] / "task.json").read_text()
            )
            images = [
                str(root / "episodes" / task["task_id"] / "goal" / f"view-{name}.png")
                for name in ("iso", "top", "front")
            ]
            if args.profile != "renders":
                from cad_env.freecad_v2.drawings import audit_pair

                drawing = root / "episodes" / task["task_id"] / "drawings"
                audit_pair(drawing)
                manifest = json.loads(
                    (drawing / args.profile / "manifest.json").read_text()
                )
                images = [
                    str(drawing / args.profile / p) for p in manifest["vlm_images"]
                ]
            if not all(Path(p).is_file() for p in images):
                raise ValueError("Missing goal render")
            env = Environment(args.output / task["task_id"], backend=backend)
            observation = {
                "state": [],
                "attempts": 0,
                "remaining_actions": 128,
                "error": None,
            }
            reward, invalid = 0, 0
            while not env.closed:
                response = policy(
                    messages(
                        task["instruction"],
                        observation,
                        env.history,
                        image_count=len(images),
                    ),
                    images,
                )
                try:
                    action = json.loads(response)
                except ValueError:
                    action = {"op": "invalid_json"}
                observation, reward, _, info = env.step(
                    action, reference=task["actions"]
                )
                invalid += not info["execution_ok"]
                if info.get("error_type") in ("Timeout", "WorkerCrash"):
                    raise RuntimeError("Native infrastructure failure: " + str(info))
            row = {
                "task_id": task["task_id"],
                "family": task["family"],
                "difficulty": task.get("difficulty"),
                "steps": env.attempts,
                "invalid_actions": invalid,
                "task_success": bool(reward),
                "validation": info.get("validation"),
            }
            status["episodes"].append(row)
            write_json(path, status)
            print(json.dumps(row), flush=True)
        status["status"] = "complete"
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(path, status)


if __name__ == "__main__":
    main()
