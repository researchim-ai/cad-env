"""Machine-readable stage gates for the FreeCAD learning process."""

import argparse
import importlib.metadata
import json
import shlex
import statistics
import sys
from collections import Counter
from pathlib import Path

from .audit import audit_dataset
from .metrics import paired_comparison


def dataset_readiness(
    root,
    *,
    min_train=200,
    min_validation=30,
    min_test=30,
    min_groups=80,
    require_feedback=True,
):
    root = Path(root).resolve()
    audit = audit_dataset(root)
    errors = list(audit["errors"])
    summary = json.loads((root / "summary.json").read_text())
    manifest = [
        json.loads(l)
        for l in (root / "manifest.jsonl").read_text().splitlines()
        if l.strip()
    ]
    accepted = [r for r in manifest if r["accepted"]]
    by_split = Counter(r["split"] for r in accepted)
    groups = {r["split_group"] for r in accepted}
    split_groups = {
        split: len({r["split_group"] for r in accepted if r["split"] == split})
        for split in ("train", "validation", "test")
    }
    for split, minimum in [
        ("train", min_train),
        ("validation", min_validation),
        ("test", min_test),
    ]:
        if by_split[split] < minimum:
            errors.append(f"{split}: {by_split[split]} accepted tasks, need {minimum}")
    if len(groups) < min_groups:
        errors.append(f"Only {len(groups)} lineage groups, need {min_groups}")
    if summary["rejected"]:
        errors.append("Oracle failures must be investigated before training")
    if not summary.get("holdout_families"):
        errors.append("An entire-family holdout is required")
    structural = Counter()
    family_counts = Counter()
    for row in accepted:
        task = json.loads(
            (root / "episodes" / row["task_id"] / "task.json").read_text()
        )
        if row["split"] == "train":
            signature = tuple(
                (a["op"], len(a.get("circles", []))) for a in task["reference_actions"]
            )
            structural[signature] += 1
            family_counts[task["family"]] += 1
        if not row.get("validation", {}).get("task_success"):
            errors.append("Accepted episode lacks task success: " + row["task_id"])
    feedback_count = summary["counts"].get("train.feedback", 0)
    recovery = 0
    feedback_path = root / "exports/train.feedback.jsonl"
    if require_feedback:
        if not feedback_path.is_file() or not feedback_count:
            errors.append("Current-state visual next-action SFT data is missing")
        elif feedback_path.is_file():
            for line in feedback_path.read_text().splitlines():
                row = json.loads(line)
                if (
                    row.get("observation_kind") != "current_state"
                    or len(row.get("images", [])) != 1
                ):
                    errors.append(
                        "Feedback rows must contain exactly one current-state view"
                    )
                recovery += int(row.get("recovery", False))
            if not recovery:
                errors.append("No executed-error recovery examples")
    if len(structural) < 3:
        errors.append("Need at least three distinct training construction structures")
    return {
        "ready": not errors,
        "errors": errors,
        "audit": audit,
        "counts": dict(by_split),
        "lineage_groups": len(groups),
        "lineage_groups_by_split": split_groups,
        "validation_promotion_sample_ready": split_groups["validation"] >= 20,
        "validation_promotion_note": "Checkpoint improvement requires at least 20 independent validation groups and a positive paired confidence interval; pilot readiness alone does not satisfy this gate.",
        "train_construction_structures": len(structural),
        "train_families": dict(family_counts),
        "visual_next_action_rows": feedback_count,
        "recovery_rows": recovery,
        "scope": "Readiness for a bounded CAD-tools pilot; four procedural families do not establish general CAD competence.",
    }


def grpo_readiness(
    scores,
    *,
    min_groups=20,
    min_valid_rate=0.8,
    min_mixed_fraction=0.1,
    max_zero_variance=0.9,
):
    groups = {}
    for row in scores:
        groups.setdefault(row["task_id"], []).append(row)
    errors = []
    if len(groups) < min_groups:
        errors.append(f"Need {min_groups} calibration prompts, got {len(groups)}")
    if any(len(rows) < 2 for rows in groups.values()):
        errors.append(
            "Every calibration prompt needs multiple independent policy samples"
        )
    valid = sum(
        r.get("shape_valid", r.get("execution_ok", False)) for r in scores
    ) / max(1, len(scores))
    mixed = sum(
        any(r["task_success"] for r in rows)
        and not all(r["task_success"] for r in rows)
        for rows in groups.values()
    ) / max(1, len(groups))
    zero = sum(
        statistics.pstdev(r["reward"] for r in rows) < 1e-8 for rows in groups.values()
    ) / max(1, len(groups))
    if valid < min_valid_rate:
        errors.append("Execution validity too low; repair SFT/curriculum before GRPO")
    if mixed < min_mixed_fraction:
        errors.append(
            "Too few groups contain both success and failure; adjust task difficulty"
        )
    if zero > max_zero_variance:
        errors.append("Most groups have no relative reward signal")
    return {
        "ready": not errors,
        "errors": errors,
        "calibration_prompts": len(groups),
        "execution_validity": valid,
        "mixed_success_fraction": mixed,
        "zero_variance_fraction": zero,
        "note": "Gate must use fresh samples from the actual SFT policy. Oracle/mutated programs are verifier tests, not policy readiness evidence.",
    }


def runtime_readiness(recipe, *, require_gpu=True):
    installed = {}
    errors = []
    for package, expected in recipe["versions"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[package] = actual
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    import torch

    devices = [
        {
            "name": torch.cuda.get_device_name(i),
            "vram_gib": round(
                torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
            ),
        }
        for i in range(torch.cuda.device_count())
    ]
    if require_gpu and len(devices) < recipe["hardware"]["gpus"]:
        errors.append("Required GPU count is unavailable")
    if require_gpu and any(
        d["vram_gib"] < 22 for d in devices[: recipe["hardware"]["gpus"]]
    ):
        errors.append("Profile expects approximately 24 GiB per GPU")
    return {
        "ready": not errors,
        "errors": errors,
        "versions": installed,
        "devices": devices,
        "note": "A profile is not a VRAM guarantee. A measured two-GPU smoke run must pass before the long run.",
    }


def training_plan(
    recipe,
    dataset,
    output,
    *,
    method="sft",
    smoke=False,
    python=None,
    sft_checkpoint=None,
):
    root = Path(dataset).resolve()
    out = Path(output).resolve()
    model, revision = recipe["model"], recipe["revision"]
    if method == "grpo":
        if not sft_checkpoint:
            raise ValueError("Program GRPO needs a merged whole-program SFT checkpoint")
        model = str(Path(sft_checkpoint).resolve())
        revision = None
        metadata = Path(model) / "cad-training-run.json"
        if (
            not metadata.is_file()
            or json.loads(metadata.read_text()).get("action_contract")
            != "whole_program"
        ):
            raise ValueError(
                "Program GRPO cannot follow a next-action SFT policy; use the separate multi-turn integration"
            )
    stage = recipe["sft" if method == "sft" else "grpo_program_baseline"]
    entry = "train.feedback.jsonl" if method == "sft" else "train.grpo.jsonl"
    python = python or sys.executable
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={recipe['hardware']['gpus']}",
        "-m",
        "cad_env.learning.train",
        "--method",
        method,
        "--model",
        model,
        "--dataset",
        str(root),
        "--export",
        str(root / "exports" / entry),
        "--output",
        str(out),
        "--require-cuda",
        "--freeze-vision",
        "--require-learning-signal",
        "--max-steps",
        str(10 if smoke else stage["max_steps"]),
        "--batch-size",
        str(stage["batch_size"]),
        "--gradient-accumulation",
        str(stage["gradient_accumulation"]),
        "--lora-rank",
        str(stage["lora_rank"]),
        "--lora-targets",
        recipe["sft"]["lora_targets"],
        "--max-length",
        str(stage["max_length"]),
        "--learning-rate",
        str(stage["learning_rate"]),
        "--save-steps",
        str(5 if smoke else recipe["sft"]["save_steps"]),
    ]
    if revision:
        command += ["--revision", revision]
    if method == "sft":
        command += [
            "--max-images",
            "1",
            "--max-pixels",
            str(recipe["sft"]["max_pixels"]),
        ]
    else:
        command += [
            "--num-generations",
            str(stage["num_generations"]),
            "--max-completion-length",
            str(stage["max_completion_length"]),
            "--cad-workers",
            str(stage["cad_workers"]),
            "--kl-beta",
            str(stage["kl_beta"]),
        ]
    return {
        "command": command,
        "shell": shlex.join(command),
        "method": method,
        "smoke": smoke,
        "mandatory_before_launch": [
            "dataset/runtime workflow check",
            "selected model license/revision recorded",
        ],
        "mandatory_after_smoke": [
            "No OOM/non-finite loss",
            "nonzero gradients and intended LoRA trainables",
            "checkpoint reload/resume",
            "peak memory and examples/sec measured on both GPUs",
        ],
        "note": "For GRPO, replace base model with merged SFT checkpoint and pass readiness gate. This command is the program-level baseline, not multi-turn GRPO.",
    }


def main():
    p = argparse.ArgumentParser(
        description="Stage gates for a measured CAD learning process"
    )
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--recipe", required=True)
    plan.add_argument("--dataset", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--run-output", required=True)
    plan.add_argument("--method", choices=["sft", "grpo"], default="sft")
    plan.add_argument("--smoke", action="store_true")
    plan.add_argument("--sft-checkpoint")
    check = sub.add_parser("check")
    check.add_argument("--dataset", required=True)
    check.add_argument("--recipe", required=True)
    check.add_argument("--output", required=True)
    calibrate = sub.add_parser("grpo-readiness")
    calibrate.add_argument("--scores", required=True)
    calibrate.add_argument("--output", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--revision")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    evaluate.add_argument("--final-test", action="store_true")
    evaluate.add_argument("--attempts", type=int, default=1)
    evaluate.add_argument("--max-steps", type=int, default=16)
    evaluate.add_argument("--max-new-tokens", type=int, default=256)
    evaluate.add_argument("--temperature", type=float, default=0)
    evaluate.add_argument("--state-only", action="store_true")
    evaluate.add_argument(
        "--text-model",
        action="store_true",
        help="Use a text-only architecture; independent of the state-only ablation",
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--disable-thinking", action="store_true")
    evaluate.add_argument(
        "--image-condition",
        choices=["full", "no_goal", "wrong_goal", "blank_goal"],
        default="full",
    )
    evaluate.add_argument("--max-pixels", type=int, default=65536)
    evaluate.add_argument("--max-context", type=int, default=4096)
    collect = sub.add_parser("collect")
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--model", required=True)
    collect.add_argument("--revision")
    collect.add_argument("--output", required=True)
    collect.add_argument("--prompts", type=int, default=20)
    collect.add_argument("--num-generations", type=int, default=4)
    collect.add_argument("--max-steps", type=int, default=16)
    collect.add_argument("--max-new-tokens", type=int, default=256)
    collect.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.command == "collect":
        from .agent_collection import collect_agents
        from .policy import HFPolicy

        policy = HFPolicy(
            args.model,
            revision=args.revision,
            temperature=0.8,
            max_new_tokens=args.max_new_tokens,
        )
        report = collect_agents(
            args.dataset,
            policy,
            args.output,
            prompts=args.prompts,
            num_generations=args.num_generations,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plan":
        report = training_plan(
            json.loads(Path(args.recipe).read_text()),
            args.dataset,
            args.run_output,
            method=args.method,
            smoke=args.smoke,
            sft_checkpoint=args.sft_checkpoint,
        )
    elif args.command == "check":
        recipe = json.loads(Path(args.recipe).read_text())
        report = {
            "dataset": dataset_readiness(args.dataset, **recipe["data_gate"]),
            "runtime": runtime_readiness(recipe),
        }
        report["ready"] = report["dataset"]["ready"] and report["runtime"]["ready"]
    elif args.command == "grpo-readiness":
        report = grpo_readiness(
            [
                json.loads(l)
                for l in Path(args.scores).read_text().splitlines()
                if l.strip()
            ]
        )
    elif args.command == "compare":
        report = paired_comparison(
            json.loads(Path(args.baseline).read_text()),
            json.loads(Path(args.candidate).read_text()),
        )
    else:
        from transformers import set_seed

        from .policy import HFPolicy
        from .protocol import evaluate_policy

        policy = HFPolicy(
            args.model,
            revision=args.revision,
            visual=not args.text_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_pixels=args.max_pixels,
            max_context=args.max_context,
            chat_template_kwargs={"enable_thinking": False}
            if args.disable_thinking
            else None,
        )

        def factory(seed):
            set_seed(seed)
            return policy

        report = evaluate_policy(
            args.dataset,
            factory,
            args.output,
            split=args.split,
            attempts=args.attempts,
            visual=not args.state_only,
            seed=args.seed,
            max_steps=args.max_steps,
            policy_protocol={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "max_context": args.max_context,
                **({"max_pixels": args.max_pixels} if args.max_pixels != 65536 else {}),
                **(
                    {"chat_template_kwargs": {"enable_thinking": False}}
                    if args.disable_thinking
                    else {}
                ),
            },
            model_identity=policy.identity,
            final_test=args.final_test,
            limit=args.limit,
            image_condition=args.image_condition,
        )
        print(json.dumps(report["metrics"], indent=2))
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report.get("ready", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
