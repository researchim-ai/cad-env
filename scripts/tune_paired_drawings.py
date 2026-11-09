"""One-pass, time-bounded SFT on TRAIN drawings, followed by the frozen 20-parent pilot."""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.batch import write_json
from cad_env.drawings.export import sha


def prepare_plan(paired, baseline, rows=1000):
    primary = Path(json.loads((paired / "request.json").read_text())["primary"])
    source_summary = json.loads((primary / "batch-summary.json").read_text())
    opposite = json.loads((paired / "summary.json").read_text())
    if source_summary["status"] != "complete" or opposite["status"] != "complete":
        raise ValueError("Both source stages must be complete")
    base = json.loads((baseline / "status.json").read_text())
    if base["status"] != "complete":
        raise ValueError("A complete prespecified baseline is required")
    heldout = json.loads((baseline / "selection.json").read_text())
    heldout_groups = {p["split_group"] for p in heldout}
    all_pairs = [
        json.loads(s) for s in (paired / "pairs.jsonl").read_text().splitlines()
    ]
    validation = {p["split_group"] for p in all_pairs if p["split"] != "train"}
    reverse = {r["name"]: Path(r["added_dataset"]) for r in opposite["shards"]}
    n_sources = len(source_summary["results"]) * 2
    if rows < n_sources or rows % n_sources:
        raise ValueError("Use an equal positive row budget divisible by source count")
    sources, selected = [], []
    for index, entry in enumerate(
        sorted(source_summary["results"], key=lambda r: r["name"])
    ):
        roots = [primary / entry["dataset"], reverse[entry["name"]]]
        roots.sort(
            key=lambda r: json.loads((r / "summary.json").read_text())[
                "drawing_standard"
            ]
        )
        paired_rows = []
        for root in roots:
            info = json.loads((root / "summary.json").read_text())
            export = root / "exports/train.feedback.jsonl"
            data = [json.loads(s) for s in export.read_text().splitlines() if s.strip()]
            budget, seed = rows // n_sources, 9100 + index
            if budget > len(data):
                raise ValueError("No cycling or repeated examples allowed")
            indices = list(range(len(data)))
            random.Random(seed).shuffle(indices)
            indices = indices[:budget]
            chosen = []
            for i in indices:
                row = data[i]
                task = json.loads(
                    (root / "episodes" / row["task_id"] / "task.json").read_text()
                )
                group = task["split_group"]
                if (
                    task["split"] != "train"
                    or group in validation
                    or group in heldout_groups
                ):
                    raise ValueError("Validation/evaluation lineage entered training")
                action = json.loads(row["completion"][0]["content"][0]["text"])
                chosen.append(
                    {
                        "task_id": row["task_id"],
                        "split_group": group,
                        "step_id": row["step_id"],
                        "recovery": row.get("recovery", False),
                        "completion": row["completion"],
                        "operation": action["op"],
                        "counterbore": "CB" in action.get("name", "")
                        or "CB" in action.get("profile", ""),
                    }
                )
                selected.append(
                    {
                        **chosen[-1],
                        "standard": info["drawing_standard"],
                        "source": str(root),
                        "row_index": i,
                    }
                )
            paired_rows.append(chosen)
            sources.append(
                {
                    "root": str(root),
                    "standard": info["drawing_standard"],
                    "rows": budget,
                    "seed": seed,
                    "selected_indices": indices,
                    "summary_sha256": sha(root / "summary.json"),
                    "export_sha256": sha(export),
                }
            )
        if paired_rows[0] != paired_rows[1]:
            raise ValueError("ISO/GOST selected different causal steps or teachers")
    if (
        len(selected) != rows
        or len({(r["source"], r["row_index"]) for r in selected}) != rows
    ):
        raise ValueError("Wrong number of unique training examples")
    return {
        "sources": sources,
        "selected": selected,
        "training_rows": rows,
        "standard_rows": dict(Counter(r["standard"] for r in selected)),
        "unique_train_parents": len({r["split_group"] for r in selected}),
        "operation_counts": dict(Counter(r["operation"] for r in selected)),
        "counterbore_rows": sum(r["counterbore"] for r in selected),
        "all_validation_groups_excluded": len(validation),
        "frozen_evaluation_groups_excluded": len(heldout_groups),
        "baseline_selection_sha256": sha(baseline / "selection.json"),
        "sampling": "Unique TRAIN rows only; same row indices/seed and teacher actions in both representations; no source cycling",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--paired", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--parent-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--train-minutes", type=float, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--reuse-prepared", action="store_true")
    p.add_argument("--resume-paused", action="store_true")
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not args.prepare_only:
        raise ValueError("Physical GPU1 only")
    paired, baseline, out = (
        Path(args.paired).resolve(),
        Path(args.baseline).resolve(),
        Path(args.output).resolve(),
    )
    if args.reuse_prepared or args.resume_paused:
        state = json.loads((out / "experiment.json").read_text())
        expected_status = "paused" if args.resume_paused else "prepared"
        if state["status"] != expected_status or state["arguments"] != {
            k: v
            for k, v in vars(args).items()
            if k not in ("prepare_only", "reuse_prepared", "resume_paused")
        }:
            raise ValueError("Prepared experiment arguments differ")
        plan = json.loads((out / "sampling-plan.json").read_text())
        if sha(out / "sampling-plan.json") != state["sampling_plan_sha256"]:
            raise ValueError("Prepared sampling plan changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        plan = prepare_plan(paired, baseline, args.rows)
        write_json(out / "sampling-plan.json", plan)
        state = {
            "status": "prepared",
            "arguments": {
                k: v
                for k, v in vars(args).items()
                if k not in ("prepare_only", "reuse_prepared", "resume_paused")
            },
            "sampling_plan_sha256": sha(out / "sampling-plan.json"),
            "baseline_analysis_sha256": sha(baseline / "analysis.json"),
            "epoch_cap": 1,
            "optimizer_budget_minutes": args.train_minutes,
            "selection_rule": "Final saved checkpoint at one pass or wall-time ceiling; no selection on evaluation scores",
        }
        write_json(out / "experiment.json", state)
        shutil.copyfile(__file__, out / "runner.py")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    k: plan[k]
                    for k in (
                        "training_rows",
                        "standard_rows",
                        "unique_train_parents",
                        "operation_counts",
                        "counterbore_rows",
                    )
                },
                indent=2,
            )
        )
        return

    remaining_minutes = args.train_minutes
    resume_checkpoint = None
    training_out = out / ("sft-resumed" if args.resume_paused else "sft")
    if args.resume_paused:
        pause = state["pause"]
        resume_checkpoint = Path(pause["resume_checkpoint"])
        saved = json.loads((resume_checkpoint / "trainer_state.json").read_text())
        if saved["global_step"] != pause["saved_step"]:
            raise ValueError("Resume checkpoint step differs from pause record")
        for name in (
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            if not (resume_checkpoint / name).is_file():
                raise ValueError("Incomplete resume checkpoint: " + name)
        elapsed = pause["last_observed_progress"]["elapsed_seconds"]
        remaining_minutes -= elapsed / 60
        if remaining_minutes <= 0:
            raise ValueError("Original training time budget exhausted")
        state.setdefault("resumptions", []).append(
            {
                "checkpoint": str(resume_checkpoint),
                "remaining_optimizer_minutes": remaining_minutes,
                "previous_pause": pause,
            }
        )
        for name in (
            "experiment.json",
            "runner.py",
            "training.log",
            "sft/cad-training-run.json",
            "sft/budget-progress.json",
        ):
            source = out / name
            if source.exists():
                shutil.copyfile(
                    source, source.with_name(source.name + ".before-resume")
                )
    shutil.copyfile(__file__, out / "runner.py")
    state["software_sha256"] = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path("cad_env/learning/train_modern.py"),
            Path("cad_env/learning/train_optimized.py"),
            Path("cad_env/learning/fused_numeric.py"),
        ]
    }

    def stage(name, command, env=None):
        state["status"] = name
        state["current_command"] = list(map(str, command))
        write_json(out / "experiment.json", state)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    try:
        effective = args.batch_size * args.gradient_accumulation
        steps = math.ceil(args.rows / effective)
        command = [
            sys.executable,
            "-m",
            "cad_env.learning.train_optimized",
            "--model",
            args.parent_model,
            "--dataset",
            plan["sources"][0]["root"],
            "--output",
            str(training_out),
            "--max-steps",
            str(steps),
            "--epoch-cap",
            "1",
            "--max-train-minutes",
            str(remaining_minutes),
            "--save-steps",
            "25",
            "--save-total-limit",
            "4",
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation",
            str(args.gradient_accumulation),
            "--learning-rate",
            "1e-5",
            "--lr-scheduler-type",
            "constant_with_warmup",
            "--warmup-steps",
            "10",
            "--max-length",
            "32768",
            "--max-images",
            "22",
            "--max-pixels",
            "327680",
            "--numeric-loss-weight",
            "8",
            "--vision-lora",
            "--memory-preflight",
            "--liger-layers",
            "--liger-profile",
            "swiglu",
        ]
        for source in plan["sources"][1:]:
            command += ["--extra-dataset", source["root"]]
        if resume_checkpoint is not None:
            command += ["--resume-from-checkpoint", str(resume_checkpoint)]
        command += [
            "--source-row-budgets",
            *[str(s["rows"]) for s in plan["sources"]],
            "--source-seeds",
            *[str(s["seed"]) for s in plan["sources"]],
        ]
        train_env = os.environ.copy()
        train_env["PYTHONPATH"] = os.pathsep.join(
            [
                str(Path("artifacts/runtimes/causal-conv-study/package").resolve()),
                str(Path("artifacts/runtimes/liger-study/package").resolve()),
                str(Path.cwd()),
            ]
        )
        stage("training", command, train_env)
        trained = json.loads((training_out / "cad-training-run.json").read_text())
        if (
            trained["status"] != "complete"
            or not trained["optimizer_steps"]
            or trained["epochs_completed"] > 1.00001
        ):
            raise ValueError("Incomplete/invalid training or epoch-cap violation")
        baseline_identity = json.loads((baseline / "status.json").read_text())[
            "models"
        ]["release-candidate"]["identity"]
        if trained["model"] != baseline_identity:
            raise ValueError(
                "Training did not start from the checkpoint used in the baseline"
            )
        clean_env = os.environ.copy()
        clean_env["PYTHONPATH"] = str(Path.cwd())
        stage(
            "merging",
            [
                sys.executable,
                "-m",
                "cad_env.learning.merge",
                "--adapter",
                str(training_out),
                "--output",
                str(out / "merged"),
            ],
            clean_env,
        )
        stage(
            "evaluating",
            [
                sys.executable,
                "scripts/benchmark_paired_drawings.py",
                "--paired",
                str(paired),
                "--output",
                str(out / "evaluation"),
                "--model",
                "tuned-iso-gost=" + str(out / "merged"),
                "--count",
                "20",
                "--batch-size",
                "2",
                "--workers",
                "4",
                "--stagnant-limit",
                "6",
                "--max-new-tokens",
                "1024",
            ],
            clean_env,
        )
        if sha(out / "evaluation/selection.json") != plan["baseline_selection_sha256"]:
            raise ValueError("Post-training evaluation changed the frozen 20 parents")
        shutil.copyfile(
            baseline / "oracle-controls.json", out / "evaluation/oracle-controls.json"
        )
        write_json(
            out / "evaluation/oracle-controls-provenance.json",
            {
                "reused_from": str(baseline / "oracle-controls.json"),
                "sha256": sha(baseline / "oracle-controls.json"),
            },
        )
        stage(
            "analysis",
            [
                sys.executable,
                "scripts/analyze_paired_benchmark.py",
                "--run",
                str(out / "evaluation"),
            ],
            clean_env,
        )
        before = json.loads((baseline / "analysis.json").read_text())["models"][
            "release-candidate"
        ]
        after = json.loads((out / "evaluation/analysis.json").read_text())["models"][
            "tuned-iso-gost"
        ]
        comparison = {
            "status": "complete",
            "independent_validation_parents": 20,
            "before": before,
            "after": after,
            "same_selection_sha256": plan["baseline_selection_sha256"],
            "training_rows": args.rows,
            "unique_train_parents": plan["unique_train_parents"],
            "optimizer_steps": trained["optimizer_steps"],
            "epochs_completed": trained["epochs_completed"],
            "walltime_budget": trained.get("walltime_budget"),
            "scope": "One bounded SFT pilot on TRAIN drawings. Same held-out 20 parents/protocol; no final TEST claim.",
        }
        write_json(out / "comparison.json", comparison)
        state.update(status="complete", comparison=str(out / "comparison.json"))
        write_json(out / "experiment.json", state)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "experiment.json", state)
        raise


if __name__ == "__main__":
    main()
