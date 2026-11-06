"""One bounded visual/repair SFT pilot, with matched batched before/after checks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def validate_recipe(config):
    if not 1 <= config["steps"] <= 200 or not 0 < config["train_minutes"] <= 45:
        raise ValueError("Pilot must stay within 200 steps and 45 optimizer minutes")
    if not 0 < config["total_minutes"] <= 120:
        raise ValueError("Pilot must stay within two hours including evaluation")
    if config["save_steps"] < 1 or config["save_steps"] > config["steps"]:
        raise ValueError("Intermediate checkpoint interval required")
    roots = [r["dataset"] for r in config["sources"]]
    if len(roots) != len(set(roots)):
        raise ValueError("Unique source roots required")
    budgets = [r["rows"] for r in config["sources"]]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in budgets):
        raise ValueError("Positive source row budgets required")
    if config["steps"] * 8 > sum(budgets):
        raise ValueError("This pilot must use at most one prepared-mixture epoch")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recipe", required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU 1 only")
    config = json.loads(Path(args.recipe).read_text())
    validate_recipe(config)
    out = Path(config["output"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Fresh pilot output required")
    out.mkdir(parents=True)
    (out / "recipe.json").write_text(json.dumps(config, indent=2))
    frozen = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("cad_env").rglob("*.py")
    }
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    deadline = time.monotonic() + config["total_minutes"] * 60
    state = {
        "status": "preflight",
        "steps": {},
        "test_opened": False,
        "optimizer_run": False,
    }

    def persist(**kw):
        state.update(kw)
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / "status.json")

    def read(path):
        return json.loads(Path(path).read_text())

    def run(name, module, *argv):
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in frozen.items()
        ):
            raise RuntimeError("Pilot source drift")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Total pilot budget exhausted")
        command = [sys.executable, "-m", module, *map(str, argv)]
        persist(status=name)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=True,
            )
        state["steps"][name] = {"status": "complete", "command": command}
        persist()

    def evaluate(name, model, dataset, steps=8, condition="full"):
        target = out / name
        run(
            name,
            "cad_env.learning.batch_evaluate",
            "--model",
            model,
            "--dataset",
            dataset,
            "--output",
            target,
            "--batch-size",
            4,
            "--workers",
            8,
            "--max-steps",
            steps,
            "--image-condition",
            condition,
        )
        report = read(target / "evaluation.json")
        if report["status"] != "complete" or report.get("limited_evaluation"):
            raise RuntimeError("Incomplete benchmark")
        return report

    try:
        persist()
        from .train import check_training_source
        from .visual_audit import audit_visual

        for source in config["sources"]:
            if source["dataset"] == config["repairs"]:
                continue
            root = Path(source["dataset"])
            check_training_source(root, root / "exports/train.feedback.jsonl")
        if not audit_visual(config["visual_dataset"])["ok"]:
            raise RuntimeError("V6 visual audit failed")
        baseline = evaluate(
            "baseline_visual", config["parent"], config["visual_dataset"]
        )
        run(
            "collect_train_errors",
            "cad_env.learning.visual_bandit",
            "--dataset",
            config["visual_dataset"],
            "--model",
            config["parent"],
            "--output",
            out / "calibration",
            "--prompts",
            32,
            "--generations",
            4,
        )
        run(
            "build_repairs",
            "cad_env.learning.onpolicy_repairs",
            "--dataset",
            config["visual_dataset"],
            "--calibration",
            out / "calibration",
            "--output",
            config["repairs"],
            "--workers",
            4,
            "--max-parents",
            16,
        )
        repair = read(Path(config["repairs"]) / "repair-generation.json")
        if repair["status"] != "complete" or not repair["tasks"]:
            raise RuntimeError(
                "No verified real policy mistakes; do not fabricate repair supervision"
            )
        persist(repair_generation=repair)
        train, merged = out / "sft", out / "merged"
        argv = [
            "--model",
            config["parent"],
            "--dataset",
            config["sources"][0]["dataset"],
            "--output",
            train,
            "--max-steps",
            config["steps"],
            "--max-train-minutes",
            config["train_minutes"],
            "--save-steps",
            config["save_steps"],
            "--gradient-accumulation",
            8,
            "--batch-size",
            1,
            "--learning-rate",
            1e-5,
            "--numeric-loss-weight",
            8,
            "--max-length",
            8192,
            "--max-pixels",
            327680,
            "--max-images",
            4,
            "--vision-lora",
            "--source-row-budgets",
            *[s["rows"] for s in config["sources"]],
        ]
        for source in config["sources"][1:]:
            argv += ["--extra-dataset", source["dataset"]]
        run("short_sft", "cad_env.learning.train_modern", *argv)
        training = read(train / "cad-training-run.json")
        if training["status"] != "complete":
            raise RuntimeError("Incomplete training")
        persist(
            optimizer_run=True,
            optimizer_steps=training["optimizer_steps"],
            walltime_budget=training["walltime_budget"],
        )
        run("merge", "cad_env.learning.merge", "--adapter", train, "--output", merged)
        candidate = evaluate("candidate_visual", merged, config["visual_dataset"])
        from .visual_diagnostics import compare_slices

        slices = compare_slices(config["visual_dataset"], baseline, candidate)
        (out / "visual-slices.json").write_text(json.dumps(slices, indent=2))
        from .metrics import paired_comparison

        comparisons = {"visual": paired_comparison(baseline, candidate)}
        reports = {}
        for name, dataset, reference, budget in config["retention"]:
            report = evaluate("candidate_" + name, merged, dataset, budget)
            parent = read(reference)
            if parent.get("execution_profile") != report.get("execution_profile"):
                raise ValueError("Parent/candidate batch execution profiles differ")
            comparisons[name] = paired_comparison(parent, report)
            reports[name] = report
            persist(comparisons=comparisons)
        from .control_metrics import paired_image_control

        controls = {}
        for condition in ["no_goal", "blank_goal", "wrong_goal"]:
            control = evaluate(
                condition, merged, config["grounding_dataset"], condition=condition
            )
            controls[condition] = paired_image_control(reports["grounding"], control)
            persist(image_controls=controls)
        eligible = comparisons["visual"]["improvement_supported"] and all(
            comparisons[name]["success_at_1_delta"] >= -0.05
            for name in ["simple", "complex", "entities"]
        )
        persist(
            status="complete",
            comparisons=comparisons,
            validation_gates_passed=eligible,
            interpretation="One exploratory mixed-curriculum pilot, not a causal ablation. V6 validation has 17 physical parents. TEST stays sealed; no further training queued.",
        )
    except Exception as exc:
        persist(
            status="failed",
            error=str(exc),
            note="No automatic retry or budget extension; inspect preserved checkpoints and completed stages",
        )
        raise


if __name__ == "__main__":
    main()
