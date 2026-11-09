"""Finish the release trial with raw/unlabelled retention inside its total budget.

Runs only after the main trial releases GPU1. Uses existing evaluator/model
contracts; never changes the running trial's frozen sources or checkpoint.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU1 only")
    root = Path(a.trial).resolve()
    out = root / "raw-retention"
    out.mkdir(exist_ok=a.resume)
    if (
        a.resume
        and json.loads((out / "status.json").read_text())["status"]
        != "waiting_for_main_trial"
    ):
        raise ValueError("Only an interrupted waiter can be resumed")
    state = {
        "status": "waiting_for_main_trial",
        "promoted": False,
        "test_opened": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Additional unlabelled visual retention; shares main 165-minute execution deadline. No training.",
    }

    def persist():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / "status.json")

    persist()
    try:
        while True:
            main_state = json.loads((root / "status.json").read_text())
            if main_state["status"] == "failed":
                raise RuntimeError("Main trial failed: no automatic continuation")
            freeze = root / "source-freeze.json"
            deadline = (
                freeze.stat().st_mtime + main_state["limits"]["total_minutes"] * 60
                if freeze.exists()
                else None
            )
            deadline = main_state.get("recovery", {}).get(
                "execution_deadline_unix", deadline
            )
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Shared release-trial deadline exhausted")
            if main_state["status"] == "complete":
                break
            time.sleep(15)
        baseline = json.loads(
            Path(
                "artifacts/grounding-ablation-v1/auxiliary/native/evaluation.json"
            ).read_text()
        )
        from cad_env.learning.metrics import paired_comparison

        for arm in ["control", "candidate"]:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("No remaining budget for raw retention")
            command = [
                sys.executable,
                "-m",
                "cad_env.learning.batch_evaluate",
                "--model",
                str(root / arm / "merged"),
                "--dataset",
                "artifacts/hard-unlabelled-v1",
                "--output",
                str(out / arm),
                "--batch-size",
                "4",
                "--workers",
                "8",
                "--max-steps",
                "8",
            ]
            state.update(status=arm, command=command)
            persist()
            with (out / (arm + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=remaining,
                )
            report = json.loads((out / arm / "evaluation.json").read_text())
            state.setdefault("comparisons", {})[arm] = paired_comparison(
                baseline, report
            )
            state.setdefault("metrics", {})[arm] = report["metrics"]
            persist()
        state["candidate_raw_point_retained"] = (
            state["comparisons"]["candidate"]["success_at_1_delta"] >= 0
        )
        state["all_pilot_gates_passed"] = (
            bool(main_state.get("qualified_for_review"))
            and state["candidate_raw_point_retained"]
        )
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
