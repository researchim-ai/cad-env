"""One bounded harder-curriculum pilot, using a verified throughput profile."""

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
    p.add_argument("--profile-report", required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU 1 only")
    profile = json.loads(Path(args.profile_report).read_text())
    mode = profile["recommended"]
    if not mode or not profile["results"][mode]["eligible"]:
        raise ValueError("No verified faster profile")
    batch = 4 if mode.endswith("batch4") else 2 if mode.endswith("batch2") else 1
    # New long-history rows differ from the throughput sample. Keep batch <=2
    # and require a forward/backward memory probe on their longest rows.
    if batch > 2:
        raise ValueError("Batch4 needs a separate longest-row qualification")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 180 * 60
    parent = "artifacts/visual-followup-500-fast-v1/merged"
    datasets = {
        "hard_visual": "artifacts/hard-unlabelled-v1",
        "hard_construction": "artifacts/hard-construction-v2",
    }
    frozen = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("cad_env").rglob("*.py")
    }
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {
        "status": "starting",
        "test_opened": False,
        "profile": mode,
        "stages": {},
        "comparisons": {},
        "limits": {
            "optimizer_minutes": 45,
            "total_minutes": 180,
            "max_steps": 300,
            "effective_batch": 8,
        },
    }

    def persist(**kwargs):
        state.update(kwargs)
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    def read(p):
        return json.loads(Path(p).read_text())

    def run(name, module, *argv):
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in frozen.items()
        ):
            raise RuntimeError("Source drift")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Total hard-pilot deadline exhausted")
        command = [sys.executable, "-m", module, *map(str, argv)]
        env = os.environ.copy()
        if module == "cad_env.learning.train_optimized" and "causal" in mode:
            env["PYTHONPATH"] = (
                str(Path("artifacts/runtimes/causal-conv-study/package").resolve())
                + os.pathsep
                + env.get("PYTHONPATH", ".")
            )
        persist(status=name)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
                env=env,
            )
        state["stages"][name] = {"status": "complete", "command": command}
        persist()

    def evaluate(name, model, dataset, steps=8, condition="full"):
        run(
            name,
            "cad_env.learning.batch_evaluate",
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
        report = read(out / name / "evaluation.json")
        if report["status"] != "complete" or report.get("limited_evaluation"):
            raise RuntimeError("Incomplete validation")
        return report

    try:
        persist()
        from .train import check_training_source

        for dataset in datasets.values():
            if not read(Path(dataset) / "hard-audit.json")["ok"]:
                raise ValueError("Hard corpus audit failed")
            check_training_source(
                Path(dataset), Path(dataset) / "exports/train.feedback.jsonl"
            )
        baseline = {
            name: evaluate(
                "baseline_" + name,
                parent,
                dataset,
                24 if name == "hard_construction" else 8,
            )
            for name, dataset in datasets.items()
        }
        # This collector is explicitly single-mark TRAIN calibration on old V6,
        # not mislabeled as sequential RL or applied to unsupported multi-edit tasks.
        run(
            "collect_train_errors",
            "cad_env.learning.visual_bandit",
            "--dataset",
            "artifacts/visual-grounding-v6-units",
            "--model",
            parent,
            "--output",
            out / "calibration",
            "--prompts",
            16,
            "--generations",
            4,
        )
        run(
            "build_repairs",
            "cad_env.learning.onpolicy_repairs",
            "--dataset",
            "artifacts/visual-grounding-v6-units",
            "--calibration",
            out / "calibration",
            "--output",
            out / "repairs",
            "--workers",
            4,
            "--max-parents",
            8,
        )
        repair = read(out / "repairs/repair-generation.json")
        sources = [
            (datasets["hard_visual"], 1000),
            (datasets["hard_construction"], 600),
            (
                str(out / "repairs")
                if repair["status"] == "complete" and repair["tasks"]
                else "artifacts/visual-grounding-v3",
                200,
            ),
            ("artifacts/complex-cad-v3", 200),
            ("artifacts/gpu1-corpus-v2", 200),
            ("artifacts/visual-grounding-v6-units", 200),
        ]
        if sum(n for _, n in sources) != 300 * 8:
            raise ValueError("Exactly one prepared-mixture epoch required")
        recipe = {
            "parent": parent,
            "sources": sources,
            "steps": 300,
            "batch_size": batch,
            "gradient_accumulation": 8 // batch,
            "profile_report": str(Path(args.profile_report).resolve()),
            "profile": mode,
            "repairs": repair,
            "scope": "One exploratory mixed harder curriculum, not a causal ablation; at most one prepared-mixture epoch; no TEST or retry",
        }
        (out / "recipe.json").write_text(json.dumps(recipe, indent=2))
        persist(repair_generation=repair)
        argv = [
            "--model",
            parent,
            "--dataset",
            sources[0][0],
            "--output",
            out / "sft",
            "--max-steps",
            300,
            "--max-train-minutes",
            45,
            "--save-steps",
            100,
            "--save-total-limit",
            3,
            "--batch-size",
            batch,
            "--gradient-accumulation",
            8 // batch,
            "--learning-rate",
            5e-6,
            "--lr-scheduler-type",
            "constant_with_warmup",
            "--warmup-steps",
            10,
            "--numeric-loss-weight",
            8,
            "--max-length",
            8192,
            "--max-images",
            4,
            "--max-pixels",
            327680,
            "--vision-lora",
            "--memory-preflight",
            "--source-row-budgets",
            *[n for _, n in sources],
        ]
        for dataset, _ in sources[1:]:
            argv += ["--extra-dataset", dataset]
        if "liger" in mode:
            argv += ["--liger-layers", "--liger-profile", "swiglu"]
        if "nogc" in mode:
            argv += ["--no-gradient-checkpointing"]
        if "hybrid" in mode:
            argv += ["--checkpoint-policy", "linear"]
        train_dir = out / "sft"
        try:
            run("sft", "cad_env.learning.train_optimized", *argv)
        except subprocess.CalledProcessError:
            failed = read(train_dir / "cad-training-run.json")
            message = failed.get("error", "").lower()
            memory_failure = (
                "out of memory" in message or "memory probe lacks" in message
            )
            if (
                not memory_failure
                or (train_dir / "budget-progress.json").exists()
                or (batch == 1 and not any(p in mode for p in ["nogc", "hybrid"]))
            ):
                raise
            # One bounded fallback is allowed only before any optimizer update.
            # New longer rows can exceed the short throughput corpus's envelope.
            persist(
                memory_fallback={
                    "reason": message,
                    "optimizer_steps_before_fallback": 0,
                    "actual_batch": 1,
                    "gradient_checkpointing": True,
                }
            )
            train_dir = out / "sft-memory-fallback"
            argv[argv.index("--output") + 1] = train_dir
            argv[argv.index("--batch-size") + 1] = 1
            argv[argv.index("--gradient-accumulation") + 1] = 8
            if "--no-gradient-checkpointing" in argv:
                argv.remove("--no-gradient-checkpointing")
            if "--checkpoint-policy" in argv:
                index = argv.index("--checkpoint-policy")
                del argv[index : index + 2]
            run("sft_memory_fallback", "cad_env.learning.train_optimized", *argv)
        training = read(train_dir / "cad-training-run.json")
        if training["status"] != "complete":
            raise RuntimeError("Incomplete optimizer run")
        persist(
            optimizer_steps=training["optimizer_steps"],
            completed_planned_steps=training["completed_planned_steps"],
        )
        run(
            "merge",
            "cad_env.learning.merge",
            "--adapter",
            train_dir,
            "--output",
            out / "merged",
        )
        from .metrics import paired_comparison

        comparisons = {}
        for name, dataset in datasets.items():
            candidate = evaluate(
                "candidate_" + name,
                out / "merged",
                dataset,
                24 if name == "hard_construction" else 8,
            )
            comparisons[name] = paired_comparison(baseline[name], candidate)
            persist(comparisons=comparisons)
        retention = [
            (
                "fresh",
                "artifacts/visual-followup-500-fast-v1/fresh-validation",
                "fresh_candidate",
                8,
            ),
            ("simple", "artifacts/gpu1-corpus-v2", "candidate_simple", 16),
            ("complex", "artifacts/complex-cad-v3", "candidate_complex", 16),
            ("grounding", "artifacts/visual-grounding-v1", "candidate_grounding", 8),
            (
                "entities",
                "artifacts/visual-grounding-v4-entities",
                "candidate_entities",
                8,
            ),
        ]
        controls = {}
        for name, dataset, reference, steps in retention:
            candidate = evaluate("retention_" + name, out / "merged", dataset, steps)
            comparisons[name] = paired_comparison(
                read(
                    Path("artifacts/visual-followup-500-fast-v1")
                    / reference
                    / "evaluation.json"
                ),
                candidate,
            )
            persist(comparisons=comparisons)
            if name == "grounding":
                from .control_metrics import paired_image_control

                for condition in ["no_goal", "blank_goal", "wrong_goal"]:
                    control = evaluate(
                        condition, out / "merged", dataset, steps, condition
                    )
                    controls[condition] = paired_image_control(candidate, control)
                    persist(image_controls=controls)
        passed = (
            comparisons["hard_visual"]["improvement_supported"]
            and comparisons["hard_construction"]["success_at_1_delta"] >= 0
            and all(
                comparisons[name]["success_at_1_delta"] >= -0.05
                for name, _, _, _ in retention
            )
            and all(c["image_effect_supported"] for c in controls.values())
        )
        persist(
            status="complete",
            validation_gates_passed=passed,
            note="No automatic promotion, TEST access or follow-on training",
        )
    except Exception as exc:
        persist(
            status="failed",
            error=str(exc),
            note="No retry; completed stages and checkpoints preserved",
        )
        raise


if __name__ == "__main__":
    main()
