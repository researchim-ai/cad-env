"""Bounded matched-update trial of verified stop/repair data versus clean SFT.

Neither arm changes the evaluation harness. Exact shared anchor rows are held
fixed with explicit source seeds. No source cycles or duplicate selected rows
are allowed. TEST remains sealed; qualification is distinct from promotion.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .feedback_normalization import normalize_feedback_prompt
from .perception_data import sha

PARENT = "artifacts/grounding-ablation-v1/auxiliary/merged"
FRESH = "artifacts/release-curriculum-v2"
OLD = "artifacts/assisted-native-v1/canonical-ids"
ANCHORS = [
    ("artifacts/assisted-native-v1/train-canonical-ids", 200, 43),
    ("artifacts/hard-construction-v2", 200, 44),
    ("artifacts/hard-unlabelled-v1", 100, 45),
    ("artifacts/gpu1-corpus-v2", 100, 46),
]


def plans():
    repairs = Path("artifacts/release-policy-repairs-v2")
    n = min(
        100,
        json.loads((repairs / "summary.json").read_text())["counts"]["train.feedback"],
    )
    if not n:
        raise ValueError("Real policy recovery data required")
    return {
        "control": [("artifacts/release-clean-v2", 1000, 42), *ANCHORS],
        "candidate": [
            ("artifacts/release-clean-v2", 600 - n, 42),
            ("artifacts/release-revisit-general-v2", 200, 47),
            ("artifacts/release-stop-contrast-v2", 200, 49),
            (str(repairs), n, 48),
            *ANCHORS,
        ],
    }


def census(sources, audit_cache=None):
    from .train import check_training_source

    seen = set()
    cache = {}
    reports = []
    total = 0
    for source, budget, seed in sources:
        root = Path(source)
        export = root / "exports/train.feedback.jsonl"
        key = (str(root.resolve()), sha(root / "summary.json"), sha(export))
        if audit_cache is None or key not in audit_cache:
            check_training_source(root, export)
            if audit_cache is not None:
                audit_cache.add(key)
        rows = list(map(json.loads, export.read_text().splitlines()))
        if budget > len(rows):
            raise ValueError("Source cycling forbidden in release trial")
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        selected = [rows[i] for i in indices[:budget]]
        ops = Counter()
        for row in selected:
            images = []
            for relative in row["images"]:
                path = root / relative
                if path not in cache:
                    cache[path] = sha(path)
                images.append(cache[path])
            key = hashlib.sha256(
                json.dumps(
                    [
                        normalize_feedback_prompt(row["prompt"]),
                        row["completion"],
                        images,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if key in seen:
                raise ValueError("Duplicate selected training row across sources")
            seen.add(key)
            ops[json.loads(row["completion"][0]["content"][0]["text"])["op"]] += 1
        total += len(selected)
        reports.append(
            {
                "root": source,
                "budget": budget,
                "seed": seed,
                "source_rows": len(rows),
                "selected_parents": len({r["split_group"] for r in selected}),
                "selected_tasks": len({r["task_id"] for r in selected}),
                "target_ops": dict(ops),
                "summary_sha256": sha(root / "summary.json"),
                "export_sha256": sha(export),
            }
        )
    if total != 1600:
        raise ValueError("Each arm must have exactly 1600 rows")
    return {"rows": total, "unique_rows": len(seen), "sources": reports}


def finish_results(out, state):
    def read_eval(name):
        report = json.loads((out / name / "evaluation.json").read_text())
        if report["status"] != "complete" or report["limited_evaluation"]:
            raise ValueError("Incomplete evaluation")
        return report

    from .metrics import paired_comparison

    for before, after in [
        ("parent_fresh", "candidate_fresh"),
        ("control_fresh", "candidate_fresh"),
        ("parent_simple", "candidate_simple"),
        ("parent_construction", "candidate_construction"),
    ]:
        state["comparisons"][before + "_vs_" + after] = paired_comparison(
            read_eval(before), read_eval(after)
        )
    baseline_old = json.loads(
        Path(
            "artifacts/assisted-native-v1/evaluation/canonical_ids/evaluation.json"
        ).read_text()
    )
    old_comparison = paired_comparison(baseline_old, read_eval("candidate_old"))
    state["comparisons"]["parent_old_vs_candidate_old"] = old_comparison
    full = read_eval("candidate_old")["metrics"]["success_at_1"]
    state["gates"] = {
        "beats_parent_on_fresh": state["comparisons"][
            "parent_fresh_vs_candidate_fresh"
        ]["improvement_supported"],
        "beats_equal_update_control_on_fresh": state["comparisons"][
            "control_fresh_vs_candidate_fresh"
        ]["improvement_supported"],
        "old_assisted_point_retained": old_comparison["success_at_1_delta"] >= 0,
        "simple_point_retained": state["comparisons"][
            "parent_simple_vs_candidate_simple"
        ]["success_at_1_delta"]
        >= 0,
        "construction_point_retained": state["comparisons"][
            "parent_construction_vs_candidate_construction"
        ]["success_at_1_delta"]
        >= 0,
        "goal_image_controls": all(
            full - read_eval("candidate_" + c)["metrics"]["success_at_1"] >= 0.25
            for c in ["blank_goal", "wrong_goal"]
        ),
    }
    state["qualified_for_review"] = all(state["gates"].values())
    state["status"] = "complete"
    state["note"] = (
        "Point retention gates are pragmatic pilot checks, not statistical equivalence or evidence of general CAD competence. TEST and external evaluation remain separate."
    )


def resume_trial(out):
    """Recover one interrupted candidate without replaying completed stages.

    Exclude inactive downtime from the original execution budget, but charge
    already executed work, including the updates lost after the checkpoint.
    """
    state = json.loads((out / "status.json").read_text())
    if state["status"] != "candidate_train":
        raise ValueError("Recovery supports interrupted candidate training only")
    freeze_path = out / "source-freeze.json"
    frozen = json.loads(freeze_path.read_text())
    for path, digest in frozen.items():
        if Path(path).resolve() != Path(__file__).resolve() and sha(path) != digest:
            raise RuntimeError("Source drift before recovery: " + path)
    progress = out / "candidate/sft/budget-progress.json"
    remaining = state["limits"]["total_minutes"] * 60 - (
        progress.stat().st_mtime - freeze_path.stat().st_mtime
    )
    if remaining <= 0:
        raise TimeoutError("No execution budget remaining")
    backup = out / "interruption-evidence"
    backup.mkdir(exist_ok=False)
    for path in [out / "status.json", freeze_path]:
        (backup / path.name).write_bytes(path.read_bytes())
    checkpoint = out / "candidate/sft/checkpoint-50"
    if json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != 50:
        raise ValueError("Expected verified checkpoint at step 50")
    state["recovery"] = {
        "checkpoint": str(checkpoint),
        "last_observed_step": json.loads(progress.read_text())["optimizer_steps"],
        "remaining_execution_seconds": remaining,
        "execution_deadline_unix": time.time() + remaining,
        "reason": "Process absent with stale running status; resume optimizer/RNG from saved checkpoint in a new output directory.",
    }
    # Only orchestration changed; trainer, kernels, observations and data frozen.
    frozen[str(Path(__file__).relative_to(Path.cwd()))] = sha(Path(__file__))
    freeze_path.write_text(json.dumps(frozen, indent=2))
    deadline = time.monotonic() + remaining

    def persist():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / "status.json")

    def run(name, command):
        if any(sha(path) != digest for path, digest in frozen.items()):
            raise RuntimeError("Source drift during recovered trial")
        if deadline <= time.monotonic():
            raise TimeoutError("Remaining execution budget exhausted")
        state["status"] = name
        state["stages"][name] = {"status": "running", "command": command}
        persist()
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=deadline - time.monotonic(),
            )
        state["stages"][name]["status"] = "complete"
        persist()

    try:
        command = list(state["stages"]["candidate_train"]["command"])
        resumed = out / "candidate/sft-resumed"
        command[command.index("--output") + 1] = str(resumed)
        command += ["--resume-from-checkpoint", str(checkpoint)]
        run("candidate_train_resumed", command)
        result = json.loads((resumed / "cad-training-run.json").read_text())
        if result["status"] != "complete" or result["optimizer_steps"] != 200:
            raise RuntimeError("Resumed candidate did not reach 200 updates")
        state["actual_steps"]["candidate"] = 200
        run(
            "candidate_merge",
            [
                sys.executable,
                "-m",
                "cad_env.learning.merge",
                "--adapter",
                str(resumed),
                "--output",
                str(out / "candidate/merged"),
            ],
        )
        evaluations = [
            (arm + "_" + name, arm, dataset, steps, "full")
            for arm in ["control", "candidate"]
            for name, dataset, steps in [
                ("fresh", FRESH, 12),
                ("old", OLD, 8),
                ("construction", "artifacts/hard-construction-v2", 24),
                ("simple", "artifacts/gpu1-corpus-v2", 16),
            ]
        ]
        evaluations += [
            ("candidate_" + condition, "candidate", OLD, 8, condition)
            for condition in ["blank_goal", "wrong_goal"]
        ]
        for name, arm, dataset, steps, condition in evaluations:
            run(
                name,
                [
                    sys.executable,
                    "-m",
                    "cad_env.learning.batch_evaluate",
                    "--model",
                    str(out / arm / "merged"),
                    "--dataset",
                    dataset,
                    "--output",
                    str(out / name),
                    "--batch-size",
                    "4",
                    "--workers",
                    "8",
                    "--max-steps",
                    str(steps),
                    "--image-condition",
                    condition,
                ],
            )
        finish_results(out, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not a.prepare_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU1 only")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=a.resume)
    if a.resume:
        resume_trial(out)
        return
    arms = plans()
    state = {
        "status": "preparing",
        "parent": PARENT,
        "test_opened": False,
        "promoted": False,
        "stages": {},
        "comparisons": {},
        "limits": {
            "total_minutes": 165,
            "optimizer_minutes_per_arm": 35,
            "steps_per_arm": 200,
            "rows_per_arm": 1600,
            "effective_batch": 8,
            "prepared_mixture_epochs": 1,
            "checkpoint_interval": 50,
        },
        "hypothesis": "Explicit correct/incorrect state contrasts plus real student continuations improve native stop/recovery versus equal-update clean-only SFT. This bundled recipe does not isolate each ingredient.",
    }

    def persist():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / "status.json")

    persist()
    try:
        audit_cache = set()
        for arm, sources in arms.items():
            state.setdefault("census", {})[arm] = census(sources, audit_cache)
            persist()
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        persist()
        raise
    state["status"] = "prepared"
    persist()
    if a.prepare_only:
        return
    frozen = {str(p): sha(p) for p in Path("cad_env").rglob("*.py")}
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    deadline = time.monotonic() + 165 * 60

    def run(name, module, argv):
        if any(sha(p) != h for p, h in frozen.items()):
            raise RuntimeError("Source drift during controlled trial")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Total trial deadline exhausted")
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

    def evaluate(name, model, dataset, steps=12, condition="full"):
        run(
            name,
            "cad_env.learning.batch_evaluate",
            [
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
            ],
        )

    def read_eval(name):
        s = json.loads((out / name / "evaluation.json").read_text())
        if s["status"] != "complete" or s["limited_evaluation"]:
            raise ValueError("Incomplete evaluation")
        return s

    try:
        evaluate("parent_fresh", PARENT, FRESH)
        evaluate("parent_construction", PARENT, "artifacts/hard-construction-v2", 24)
        evaluate("parent_simple", PARENT, "artifacts/gpu1-corpus-v2", 16)
        for arm, sources in arms.items():
            argv = [
                "--model",
                PARENT,
                "--dataset",
                sources[0][0],
                "--output",
                out / arm / "sft",
                "--source-row-budgets",
                *[s[1] for s in sources],
                "--source-seeds",
                *[s[2] for s in sources],
                "--max-steps",
                200,
                "--max-train-minutes",
                35,
                "--save-steps",
                50,
                "--save-total-limit",
                4,
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
            for source, _, _ in sources[1:]:
                argv += ["--extra-dataset", source]
            run(arm + "_train", "cad_env.learning.train_optimized", argv)
            checkpoints = list(
                (out / arm / "sft").glob("checkpoint-*/trainer_state.json")
            )
            steps = max(
                (json.loads(p.read_text())["global_step"] for p in checkpoints),
                default=0,
            )
            state.setdefault("actual_steps", {})[arm] = steps
            persist()
            if steps != 200:
                raise RuntimeError(
                    "Arm did not finish one prepared-mixture epoch; do not claim matched updates"
                )
            run(
                arm + "_merge",
                "cad_env.learning.merge",
                ["--adapter", out / arm / "sft", "--output", out / arm / "merged"],
            )
        for arm in arms:
            model = out / arm / "merged"
            evaluate(arm + "_fresh", model, FRESH)
            evaluate(arm + "_old", model, OLD, 8)
            evaluate(arm + "_construction", model, "artifacts/hard-construction-v2", 24)
            evaluate(arm + "_simple", model, "artifacts/gpu1-corpus-v2", 16)
        for condition in ["blank_goal", "wrong_goal"]:
            evaluate(
                "candidate_" + condition, out / "candidate/merged", OLD, 8, condition
            )
        finish_results(out, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
