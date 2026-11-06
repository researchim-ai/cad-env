"""One bounded learning-curve run, then paired native validation. No retries."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def training_command(config):
    sources = config["sources"]
    if sum(s["rows"] for s in sources) != 4000 or config["steps"] != 500:
        raise ValueError("Exactly one 4000-row prepared-mixture epoch required")
    if len({s["dataset"] for s in sources}) != len(sources):
        raise ValueError("Duplicate source roots")
    module = config.get("training_module", "cad_env.learning.train_modern")
    if module not in {
        "cad_env.learning.train_modern",
        "cad_env.learning.train_optimized",
    }:
        raise ValueError("Unsupported training entry point")
    batch = config.get("batch_size", 1)
    minutes = config.get("train_minutes", 105)
    if batch not in {1, 2} or not 0 < minutes <= 105:
        raise ValueError("Measured batch profile and bounded training time required")
    command = [
        sys.executable,
        "-m",
        module,
        "--model",
        config["parent"],
        "--dataset",
        sources[0]["dataset"],
        "--output",
        str(Path(config["output"]) / "sft"),
        "--max-steps",
        "500",
        "--max-train-minutes",
        str(minutes),
        "--save-steps",
        "100",
        "--save-total-limit",
        "6",
        "--gradient-accumulation",
        str(8 // batch),
        "--batch-size",
        str(batch),
        "--learning-rate",
        "1e-5",
        "--lr-scheduler-type",
        "constant_with_warmup",
        "--warmup-steps",
        "20",
        "--numeric-loss-weight",
        "8",
        "--max-length",
        "8192",
        "--max-pixels",
        "327680",
        "--max-images",
        "4",
        "--vision-lora",
        "--source-row-budgets",
        *[str(s["rows"]) for s in sources],
    ]
    for source in sources[1:]:
        command += ["--extra-dataset", source["dataset"]]
    if config.get("resume_from_checkpoint"):
        command += ["--resume-from-checkpoint", config["resume_from_checkpoint"]]
    if config.get("liger_layers"):
        if module != "cad_env.learning.train_optimized":
            raise ValueError("Liger requires the explicitly optimized entry point")
        command += ["--liger-layers"]
        command += ["--liger-profile", config.get("liger_profile", "all")]
    return command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recipe", required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU 1 only")
    config = json.loads(Path(args.recipe).read_text())
    command = training_command(config)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "recipe.json").write_text(json.dumps(config, indent=2))
    frozen = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("cad_env").rglob("*.py")
    }
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {"test_opened": False, "stages": {}, "status": "starting"}

    def persist(**kwargs):
        state.update(kwargs)
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    def run(name, argv, deadline):
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
            for p, digest in frozen.items()
        ):
            raise RuntimeError("Source drift; refusing mixed-code experiment")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Phase deadline exhausted")
        persist(status=name)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                list(map(str, argv)),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
            )
        state["stages"][name] = {"command": list(map(str, argv)), "status": "complete"}
        persist()

    def module(name, module, deadline, *argv):
        run(name, [sys.executable, "-m", module, *argv], deadline)

    def evaluate(name, model, dataset, deadline, steps=8, condition="full"):
        module(
            name,
            "cad_env.learning.batch_evaluate",
            deadline,
            "--model",
            model,
            "--dataset",
            dataset,
            "--output",
            out / name,
            "--batch-size",
            4,
            "--workers",
            8,
            "--max-steps",
            steps,
            "--image-condition",
            condition,
        )
        report = json.loads((out / name / "evaluation.json").read_text())
        if report["status"] != "complete" or report.get("limited_evaluation"):
            raise RuntimeError("Incomplete validation")
        return report

    try:
        run("training", command, time.monotonic() + 120 * 60)
        training = json.loads((out / "sft/cad-training-run.json").read_text())
        if training["status"] != "complete":
            raise RuntimeError("Training did not finish safely")
        persist(
            optimizer_steps=training["optimizer_steps"],
            training=training["walltime_budget"],
        )
        # Separate, explicitly bounded evaluation budget; no additional optimization.
        deadline = time.monotonic() + 120 * 60
        module(
            "fresh_validation",
            "cad_env.learning.visual_holdout",
            deadline,
            "--output",
            out / "fresh-validation",
        )
        parent = evaluate(
            "fresh_parent", config["parent"], out / "fresh-validation", deadline
        )
        module(
            "merge_final",
            "cad_env.learning.merge",
            deadline,
            "--adapter",
            out / "sft",
            "--output",
            out / "merged",
        )
        candidate = evaluate(
            "fresh_candidate", out / "merged", out / "fresh-validation", deadline
        )
        from .metrics import paired_comparison

        comparisons = {"fresh_validation": paired_comparison(parent, candidate)}
        persist(comparisons=comparisons)
        # Same old cohort also measures progress relative to the 160-step pilot.
        visual = evaluate(
            "candidate_visual",
            out / "merged",
            config["sources"][0]["dataset"],
            deadline,
        )
        old = json.loads(
            Path(
                "artifacts/visual-repair-pilot-v1/candidate_visual/evaluation.json"
            ).read_text()
        )
        comparisons["versus_160"] = paired_comparison(old, visual)
        for name, dataset, reference, steps in config["retention"]:
            report = evaluate(
                "candidate_" + name, out / "merged", dataset, deadline, steps
            )
            comparisons[name] = paired_comparison(
                json.loads(Path(reference).read_text()), report
            )
            persist(comparisons=comparisons)
            if name == "grounding":
                from .control_metrics import paired_image_control

                controls = {}
                for condition in ["no_goal", "blank_goal", "wrong_goal"]:
                    control = evaluate(
                        condition, out / "merged", dataset, deadline, steps, condition
                    )
                    controls[condition] = paired_image_control(report, control)
                    persist(image_controls=controls)
        for step in [100, 300]:
            checkpoint = out / "sft" / f"checkpoint-{step}"
            if not checkpoint.exists() and config.get("resume_from_checkpoint"):
                previous = Path(config["resume_from_checkpoint"])
                if previous.name == f"checkpoint-{step}":
                    checkpoint = previous
            if not checkpoint.exists():
                continue  # A wall-time stop can precede this checkpoint.
            merged = out / f"merged-{step}"
            module(
                f"merge_{step}",
                "cad_env.learning.merge",
                deadline,
                "--adapter",
                checkpoint,
                "--output",
                merged,
            )
            report = evaluate(
                f"visual_{step}", merged, config["sources"][0]["dataset"], deadline
            )
            comparisons[f"step_{step}_to_final"] = paired_comparison(report, visual)
            persist(comparisons=comparisons)
        persist(
            status="complete",
            comparisons=comparisons,
            note="Synthetic validation only; no automatic promotion, TEST opening or further training",
        )
    except Exception as exc:
        persist(
            status="failed",
            error=str(exc),
            note="No retry; checkpoints and completed stages preserved",
        )
        raise


if __name__ == "__main__":
    main()
