"""Bounded matched-update auxiliary-grounding ablation on GPU1.

Two arms start from the same frozen checkpoint, use one prepared-mixture epoch,
and differ only in replacing 224/1024 action rows with TRAIN visual questions.
Recovery/current-render evidence is deliberately excluded from both arms.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if not a.prepare_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU1 only")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = "artifacts/hard-experiment-v1/merged"
    sources = [
        "artifacts/hard-unlabelled-v1",
        "artifacts/grounding-auxiliary-v1/visual",
        "artifacts/grounding-auxiliary-v1/construction",
    ]
    from .train import check_training_source

    for source in sources:
        root = Path(source)
        check_training_source(root, root / "exports/train.feedback.jsonl")
    frozen = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("cad_env").rglob("*.py")
    }
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {
        "status": "prepared",
        "test_opened": False,
        "parent": parent,
        "limits": {
            "total_minutes": 60,
            "optimizer_minutes_per_arm": 20,
            "steps_per_arm": 128,
            "rows_per_arm": 1024,
            "effective_batch": 8,
            "prepared_mixture_epochs": 1,
        },
        "arms": {
            "control": {"budgets": [1024]},
            "auxiliary": {"budgets": [800, 192, 32]},
        },
        "stages": {},
        "comparisons": {},
        "promoted": False,
        "scope": "Matched optimizer-update control; token counts and FLOPs are not matched; no recovery rows or changed observation protocol; validation only; no automatic follow-on training",
    }

    def persist():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    persist()
    if a.prepare_only:
        return
    deadline = time.monotonic() + 60 * 60

    def run(name, module, argv):
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in frozen.items()
        ):
            raise RuntimeError(
                "Source drift: do not modify the experiment while it runs"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Total experiment deadline exhausted")
        command = [sys.executable, "-m", module, *map(str, argv)]
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
        state["stages"][name]["status"] = "complete"
        persist()

    try:
        for arm in state["arms"]:
            argv = [
                "--model",
                parent,
                "--dataset",
                sources[0],
                "--output",
                out / arm / "sft",
                "--source-row-budgets",
                *state["arms"][arm]["budgets"],
                "--max-steps",
                128,
                "--max-train-minutes",
                20,
                "--save-steps",
                32,
                "--batch-size",
                1,
                "--gradient-accumulation",
                8,
                "--learning-rate",
                5e-6,
                "--lr-scheduler-type",
                "constant_with_warmup",
                "--warmup-steps",
                10,
                "--vision-lora",
                "--numeric-loss-weight",
                8,
                "--max-length",
                8192,
                "--max-images",
                1,
                "--max-pixels",
                327680,
                "--memory-preflight",
                "--checkpoint-policy",
                "linear",
            ]
            if arm == "auxiliary":
                argv += ["--action-contract", "single_action_auxiliary"]
                for source in sources[1:]:
                    argv += ["--extra-dataset", source]
            run(arm + "_train", "cad_env.learning.train_optimized", argv)
            run(
                arm + "_merge",
                "cad_env.learning.merge",
                ["--adapter", out / arm / "sft", "--output", out / arm / "merged"],
            )
        # Trainer state records actual updates even when a walltime guard stops early.
        actual = []
        for arm in state["arms"]:
            checkpoints = list(
                (out / arm / "sft").glob("checkpoint-*/trainer_state.json")
            )
            if not checkpoints:
                raise RuntimeError("No durable checkpoint for step-count matching")
            actual.append(
                max(json.loads(p.read_text())["global_step"] for p in checkpoints)
            )
        state["actual_steps"] = dict(zip(state["arms"], actual))
        state["matched_updates"] = actual == [128, 128]
        if not state["matched_updates"]:
            raise RuntimeError(
                "Incomplete arm: do not interpret as matched-update ablation"
            )
        for arm in state["arms"]:
            run(
                arm + "_native",
                "cad_env.learning.batch_evaluate",
                [
                    "--model",
                    out / arm / "merged",
                    "--dataset",
                    sources[0],
                    "--output",
                    out / arm / "native",
                    "--batch-size",
                    4,
                    "--workers",
                    8,
                    "--max-steps",
                    8,
                ],
            )
        run(
            "component_comparison",
            "cad_env.learning.perception_probe",
            [
                "--raw-model",
                out / "control/merged",
                "--trained-model",
                out / "auxiliary/merged",
                "--output",
                out / "components",
                "--minutes",
                20,
                "--one-per-parent",
            ],
        )
        from .visual_diagnostics import compare_slices

        before = json.loads((out / "control/native/evaluation.json").read_text())
        after = json.loads((out / "auxiliary/native/evaluation.json").read_text())
        state["comparisons"]["auxiliary_vs_control"] = compare_slices(
            sources[0], before, after
        )
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
