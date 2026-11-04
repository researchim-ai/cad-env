"""Single-GPU GRPO for one visual edit followed by a fixed public finish.

This is contextual-bandit RL, not multi-turn CAD policy-gradient training.
Rewards come from fresh sampled actions executed and graded in FreeCAD.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument(
        "--mixed-prompts-only",
        action="store_true",
        help="Calibrated TRAIN curriculum for mechanical smoke runs",
    )
    p.add_argument("--resume-from-checkpoint")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Only the measured one-GPU profile is supported")
    if (
        min(args.max_steps, args.save_steps) < 1
        or args.num_generations < 2
        or args.learning_rate <= 0
    ):
        p.error("Invalid optimization budget")
    versions = {
        k: importlib.metadata.version(k)
        for k in ("torch", "transformers", "trl", "peft", "accelerate")
    }
    if any(
        versions[k] != v
        for k, v in {
            "transformers": "5.22.1",
            "trl": "1.17.1",
            "peft": "0.21.2",
        }.items()
    ):
        raise ValueError(
            "Use requirements-qwen35.lock in the isolated modern environment"
        )
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from .checkpoints import StrictResumeMixin
    from .factory import jsonl
    from .guards import make_numerical_guard
    from .policy import checkpoint_identity
    from .train import load_export
    from .train_modern import nonthinking_template
    from .visual_bandit import VisualBanditReward, bandit_prompts

    class VerifiedBanditTrainer(StrictResumeMixin, GRPOTrainer):
        pass

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    root, calibration, out = (
        Path(x).resolve() for x in (args.dataset, args.calibration, args.output)
    )
    evidence = json.loads((calibration / "calibration.json").read_text())
    identity = checkpoint_identity(args.model)
    if evidence["status"] != "complete" or not evidence["readiness"]["ready"]:
        raise ValueError(
            "Actual policy calibration has not passed the native reward-variance gate"
        )
    if evidence["model_identity"] != identity:
        raise ValueError("Calibration used different policy weights")
    if (
        evidence["dataset_sha256"]
        != hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Calibration used another corpus")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh GRPO output")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "preflight",
        "arguments": vars(args),
        "versions": versions,
        "model": identity,
        "action_contract": "single_action",
        "rl_contract": "one sampled action then fixed public finish; strict native reward",
        "calibration_sha256": hashlib.sha256(
            (calibration / "calibration.json").read_bytes()
        ).hexdigest(),
        "dataset_sha256": evidence["dataset_sha256"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(0),
        "optimizer_run": True,
        "scope": "Contextual bandit; no multi-turn policy loss or broad CAD mastery claim",
    }

    def persist():
        temporary = out / "cad-training-run.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(out / "cad-training-run.json")

    persist()
    for folder in ("agent", "learning"):
        for source in (Path(__file__).resolve().parents[1] / folder).glob("*.py"):
            target = out / "source_snapshot" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    manifest["source_sha256"] = {
        str(p.relative_to(out / "source_snapshot")): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in (out / "source_snapshot").rglob("*.py")
    }
    try:
        rows = bandit_prompts(root)
        if args.mixed_prompts_only:
            grouped = defaultdict(list)
            for line in (calibration / "scores.jsonl").read_text().splitlines():
                r = json.loads(line)
                grouped[r["task_id"]].append(r["reward"])
            mixed = {k for k, rewards in grouped.items() if min(rewards) < max(rewards)}
            rows = [r for r in rows if r["task_id"] in mixed]
        if not rows:
            raise ValueError("No eligible TRAIN prompts")
        source = out / "train.bandit.jsonl"
        jsonl(source, rows)
        dataset = load_export(source, grpo=True, dataset_root=root)
        manifest.update(
            prompts=len(rows),
            prompt_export_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        set_seed(42)
        processor = AutoProcessor.from_pretrained(
            args.model, min_pixels=3136, max_pixels=327680
        )
        processor.chat_template = nonthinking_template(processor.chat_template)
        # Validate real pixels and public prompt length, not only JSON metadata.
        for index, row in enumerate(dataset):
            text = processor.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )
            batch = processor(text=[text], images=row["images"], return_tensors="pt")
            if batch["input_ids"].shape[-1] + 128 > 8192:
                raise ValueError(
                    f"Bandit prompt {index} exceeds the fixed context budget"
                )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
            ),
        )
        trainable = {n: v for n, v in model.named_parameters() if v.requires_grad}
        if not trainable or any("visual" in n for n in trainable):
            raise ValueError("Unexpected GRPO trainable parameter set")

        def adapter_hash():
            digest = hashlib.sha256()
            for name, value in sorted(trainable.items()):
                digest.update(name.encode())
                digest.update(value.detach().cpu().float().numpy().tobytes())
            return digest.hexdigest()

        initial = adapter_hash()
        manifest["trainable_parameters"] = sum(v.numel() for v in trainable.values())
        reward = VisualBanditReward(root, out / "native-rollouts", workers=4)
        guard = make_numerical_guard(require_signal=True)
        trainer = VerifiedBanditTrainer(
            model=model,
            processing_class=processor,
            reward_funcs=reward,
            train_dataset=dataset,
            callbacks=[guard],
            args=GRPOConfig(
                output_dir=str(out),
                max_steps=args.max_steps,
                save_steps=args.save_steps,
                save_total_limit=3,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=args.num_generations,
                num_generations=args.num_generations,
                max_completion_length=128,
                temperature=0.8,
                top_p=1.0,
                top_k=0,
                beta=0.04,
                loss_type="grpo",
                scale_rewards="group",
                learning_rate=args.learning_rate,
                bf16=True,
                fp16=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                mask_truncated_completions=True,
                use_vllm=False,
                remove_unused_columns=False,
                report_to="none",
                logging_steps=1,
                logging_nan_inf_filter=False,
                seed=42,
                dataloader_num_workers=0,
            ),
        )
        manifest.update(
            status="running", generation=trainer.generation_config.to_dict()
        )
        processor.save_pretrained(out)
        persist()
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        changed = initial != adapter_hash()
        if not changed:
            raise RuntimeError("GRPO completed without changing adapter weights")
        trainer.save_model(str(out))
        processor.save_pretrained(out)
        manifest.update(
            status="complete",
            train_metrics=result.metrics,
            gradient_signal=guard.summary(),
            adapter_weights_changed=changed,
            reward_batches_with_signal=sum(v > 0 for v in reward.group_variance),
            reward_group_batches=len(reward.group_variance),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            resume_verification=getattr(trainer, "cad_resume_report", None),
        )
        persist()
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
