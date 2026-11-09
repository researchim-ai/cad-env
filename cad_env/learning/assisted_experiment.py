"""Bounded full-native qualification of assisted observations, no training."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .assisted_data import audit_assisted


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--minutes", type=float, default=45)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU1 only")
    if not 0 < a.minutes <= 45:
        p.error("At most 45 diagnostic minutes")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path("artifacts/assisted-native-v1").resolve()
    for condition in ["original-ids", "canonical-ids"]:
        report = audit_assisted(root / condition)
        if not report["ok"]:
            raise ValueError(report["errors"])
    frozen = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("cad_env").rglob("*.py")
    }
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {
        "status": "starting",
        "optimizer_updates": 0,
        "test_opened": False,
        "minutes_limit": a.minutes,
        "stages": {},
        "results": {},
        "promoted": False,
        "scope": "Full native observation-contract qualification; not a new model and not SFT improvement; same task identities and geometry, different observations",
    }

    def persist():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    persist()
    deadline = time.monotonic() + a.minutes * 60
    try:
        for name, dataset, image_condition in [
            ("original_ids", "original-ids", "full"),
            ("canonical_ids", "canonical-ids", "full"),
            ("blank_goal", "canonical-ids", "blank_goal"),
            ("wrong_goal", "canonical-ids", "wrong_goal"),
        ]:
            if any(
                hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
                for p, h in frozen.items()
            ):
                raise RuntimeError("Source drift")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Native qualification budget exhausted")
            command = [
                sys.executable,
                "-m",
                "cad_env.learning.batch_evaluate",
                "--model",
                "artifacts/grounding-ablation-v1/auxiliary/merged",
                "--dataset",
                str(root / dataset),
                "--output",
                str(out / name),
                "--batch-size",
                "4",
                "--workers",
                "8",
                "--max-steps",
                "8",
                "--image-condition",
                image_condition,
            ]
            state["status"] = name
            state["stages"][name] = {"status": "running", "command": command}
            persist()
            with (out / (name + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=remaining,
                )
            report = json.loads((out / name / "evaluation.json").read_text())
            if report["status"] != "complete" or report["limited_evaluation"]:
                raise RuntimeError("Incomplete native qualification")
            state["results"][name] = {
                "metrics": report["metrics"],
                "successes": sum(e["task_success"] for e in report["episodes"]),
                "tasks": len(report["task_ids"]),
                "dataset_digest": report["dataset_digest"],
                "protocol_digest": report["protocol_digest"],
            }
            state["stages"][name]["status"] = "complete"
            persist()
        baseline = json.loads(
            Path(
                "artifacts/grounding-ablation-v1/auxiliary/native/evaluation.json"
            ).read_text()
        )
        state["unassisted_reference"] = {
            "successes": sum(e["task_success"] for e in baseline["episodes"]),
            "tasks": len(baseline["task_ids"]),
            "note": "Prior completed same-model native result; changed observation protocols reported separately",
        }
        for name in state["results"]:
            report = json.loads((out / name / "evaluation.json").read_text())
            if report["task_ids"] != baseline["task_ids"]:
                raise RuntimeError("Task cohort mismatch")
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
