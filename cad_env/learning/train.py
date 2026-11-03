"""Optional TRL 0.24 training entrypoint. CAD generation needs none of this stack."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def load_export(path, *, grpo=False, dataset_root=None):
    from datasets import Dataset, Image, Sequence

    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Training export is empty")
    root = Path(dataset_root).resolve() if dataset_root else path.parent.parent
    keep = (
        {"prompt", "task_id", "images"} if grpo else {"prompt", "completion", "images"}
    )
    data = []
    for row in rows:
        sample = {k: v for k, v in row.items() if k in keep}
        if "images" in sample:
            # Arrow needs one content type per message column; multimodal
            # conversations use typed blocks for every role, including system.
            for key in ("prompt", "completion"):
                for message in sample.get(key, []):
                    if isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
            resolved = []
            for image in sample["images"]:
                target = (root / image).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError("Image must exist inside the dataset package")
                resolved.append(str(target))
            sample["images"] = resolved
        data.append(sample)
    dataset = Dataset.from_list(data)
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(Image()))
    return dataset


def check_text_supervision(dataset, tokenizer, max_length):
    """Fail before optimization if template boundaries or truncation corrupt labels."""
    for index, row in enumerate(dataset):
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat-template prompt/completion boundary mismatch in row {index}; choose a compatible template (including thinking-mode settings)"
            )
        if len(full_ids) > max_length:
            raise ValueError(
                f"Row {index} has {len(full_ids)} tokens, exceeding max-length={max_length}; increase the limit instead of truncating the CAD target"
            )
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"No assistant target tokens in row {index}")


def check_vision_supervision(dataset, processor, max_length, max_images=4):
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, completion_only_loss=True, max_length=None
    )
    lengths, targets = [], []
    for index, row in enumerate(dataset):
        if not 1 <= len(row["images"]) <= max_images:
            raise ValueError(f"Row {index} violates the image-count budget")
        batch = collator([row])
        length = int(batch["attention_mask"][0].sum())
        count = int((batch["labels"][0] != -100).sum())
        if length > max_length:
            raise ValueError(
                f"Visual row {index} needs {length} tokens, budget {max_length}; do not truncate image/target tokens"
            )
        if count == 0 or count >= length:
            raise ValueError(f"Incorrect completion-only mask in visual row {index}")
        lengths.append(length)
        targets.append(count)
    return {
        "rows": len(lengths),
        "max_tokens": max(lengths),
        "max_target_tokens": max(targets),
        "all_completion_masks_checked": True,
    }


def check_training_source(root, source):
    from .audit import audit_dataset

    audit = audit_dataset(root)
    if not audit["ok"]:
        raise ValueError("Dataset audit failed: " + str(audit["errors"][:3]))
    accepted = {
        r["task_id"]
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
        if (r := json.loads(line))["accepted"] and r["split"] == "train"
    }
    count = 0
    for line in Path(source).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        if row.get("task_id") not in accepted:
            raise ValueError(
                "Training input contains held-out/unregistered task: "
                + str(row.get("task_id"))
            )
    if not count:
        raise ValueError("Training input is empty")
    return audit


def main():
    parser = argparse.ArgumentParser(
        description="Train on verified CAD SFT or online GRPO rollouts (optional TRL stack)"
    )
    parser.add_argument("--method", choices=["sft", "grpo"], required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local checkpoint or an explicitly chosen Hugging Face model",
    )
    parser.add_argument("--dataset", required=True, help="Generated package root")
    parser.add_argument(
        "--export",
        help="Override training JSONL; by default exports/train.sft.jsonl or train.grpo.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--asset-root",
        help="Verified on-policy image bundle; base task splits still come from --dataset",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--require-learning-signal", action="store_true")
    parser.add_argument("--lora-targets", default="all-linear")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16, help="0 disables LoRA")
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--freecad")
    parser.add_argument("--cad-workers", type=int, default=1)
    parser.add_argument("--cad-timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.max_steps < 1
        or args.batch_size < 1
        or args.num_generations < 2
        or args.lora_rank < 0
    ):
        parser.error("Invalid training budget, batch size, generations or LoRA rank")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    generation_batch = world_size * args.batch_size * args.gradient_accumulation
    if (
        args.gradient_accumulation < 1
        or args.max_length < 1
        or args.max_completion_length < 1
        or args.save_steps < 1
    ):
        parser.error("Invalid accumulation, token or checkpoint budget")
    if args.method == "grpo" and generation_batch % args.num_generations:
        parser.error(
            "world_size * batch_size * gradient_accumulation must be divisible by num_generations"
        )
    try:
        from trl import SFTConfig, SFTTrainer

        if args.method == "grpo":
            from trl import GRPOConfig, GRPOTrainer
        import torch
        from peft import LoraConfig
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies failed to import. Use an isolated environment with requirements-learning.txt; the data factory does not require TRL."
        ) from exc
    dataset_root = Path(args.dataset).resolve()
    source = args.export or dataset_root / "exports" / (
        "train." + args.method + ".jsonl"
    )
    check_training_source(dataset_root, source)
    if importlib.metadata.version("trl") != "0.24.0":
        raise RuntimeError(
            "This launcher is tested against trl==0.24.0; use the pinned training environment"
        )
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this run profile")
    from .checkpoints import StrictResumeMixin

    class VerifiedSFTTrainer(StrictResumeMixin, SFTTrainer):
        pass

    if args.method == "grpo":

        class VerifiedGRPOTrainer(StrictResumeMixin, GRPOTrainer):
            pass

    from .policy import checkpoint_identity

    identity = checkpoint_identity(args.model, args.revision)
    previous_run=Path(args.model)/'cad-training-run.json'
    if args.method=='grpo' and previous_run.is_file() and json.loads(previous_run.read_text()).get('action_contract')=='single_action':
        raise ValueError('Program GRPO cannot consume a single-action policy; use the separately validated multi-turn adapter')
    asset_root = dataset_root
    asset_manifest = None
    if args.asset_root:
        from .bundles import verify_bundle

        asset_root = Path(args.asset_root).resolve()
        asset_manifest = verify_bundle(asset_root, source)
        parent = asset_manifest.get("parent_dataset_digest")
        if (
            parent
            != hashlib.sha256((dataset_root / "summary.json").read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Recovery supervision must be tied to this exact base dataset revision"
            )
    dataset = load_export(source, grpo=args.method == "grpo", dataset_root=asset_root)
    visual = "images" in dataset.column_names
    common = dict(
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=not args.freeze_vision,
        dataloader_pin_memory=torch.cuda.is_available(),
        learning_rate=args.learning_rate,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        seed=args.seed,
        dataloader_num_workers=0,
    )
    peft = (
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules="all-linear"
            if args.lora_targets == "all-linear"
            else args.lora_targets.split(","),
            task_type="CAUSAL_LM",
        )
        if args.lora_rank
        else None
    )
    from transformers import AutoProcessor, AutoTokenizer

    processor_kwargs = {"revision": args.revision} if args.revision else {}
    processor = (
        AutoProcessor.from_pretrained(
            args.model,
            min_pixels=56 * 56,
            max_pixels=args.max_pixels,
            **processor_kwargs,
        )
        if visual
        else AutoTokenizer.from_pretrained(args.model, **processor_kwargs)
    )
    from accelerate import PartialState

    distributed = PartialState()
    rank = distributed.process_index
    out = Path(args.output).resolve()
    output_error = None
    if distributed.is_main_process:
        if out.exists() and any(out.iterdir()) and not args.resume_from_checkpoint:
            output_error = (
                "Run output is not empty; use a new run or --resume-from-checkpoint"
            )
        else:
            out.mkdir(parents=True, exist_ok=True)
    if distributed.num_processes > 1:
        message = [output_error]
        torch.distributed.broadcast_object_list(message, src=0)
        output_error = message[0]
    if output_error:
        raise FileExistsError(output_error)
    distributed.wait_for_everyone()
    first_row = json.loads(
        next(line for line in Path(source).read_text().splitlines() if line.strip())
    )
    action_contract = "whole_program"
    if args.method == "sft":
        target = first_row["completion"][0]["content"]
        if isinstance(target, list):
            target = "".join(block.get("text", "") for block in target)
        target = json.loads(target)
        action_contract = (
            "single_action"
            if isinstance(target, dict) and "op" in target
            else "whole_program"
        )
    run_manifest = {
        "schema_version": "cad-env.training.v2",
        "status": "preparing",
        "launch_id": os.environ.get("TORCHELASTIC_RUN_ID"),
        "method": args.method,
        "action_contract": action_contract,
        "asset_bundle": asset_manifest,
        "model": identity,
        "dataset_digest": hashlib.sha256(
            (dataset_root / "summary.json").read_bytes()
        ).hexdigest(),
        "export_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "arguments": vars(args),
        "world_size": world_size,
        "effective_batch_size": generation_batch,
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "transformers", "trl", "datasets", "peft", "accelerate")
        },
        "warning": "Completion of optimization is not evidence of CAD quality; use paired native evaluation.",
    }

    def save_manifest():
        if rank == 0:
            (out / "cad-training-run.json").write_text(
                json.dumps(run_manifest, indent=2)
            )

    save_manifest()
    if rank == 0:
        import shutil

        snapshot = out / "source_snapshot"
        for folder in ("agent", "learning"):
            for file in sorted(
                (Path(__file__).resolve().parents[1] / folder).glob("*.py")
            ):
                target = snapshot / folder / file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
        run_manifest["source_sha256"] = {
            str(file.relative_to(snapshot)): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in sorted(snapshot.rglob("*.py"))
        }
        save_manifest()
        import platform
        import sys

        packages = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        }
        (out / "environment.json").write_text(
            json.dumps(
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": packages,
                    "torch_cuda": torch.version.cuda,
                },
                indent=2,
            )
        )
    try:
        supervision = {}
        if args.method == "sft":
            if visual:
                supervision = check_vision_supervision(
                    dataset, processor, args.max_length, args.max_images
                )
            else:
                check_text_supervision(dataset, processor, args.max_length)
        else:
            # GRPO must never silently drop the task or current state on the left.
            for index, row in enumerate(dataset):
                if visual:
                    inputs = processor(
                        text=[
                            processor.apply_chat_template(
                                row["prompt"],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        ],
                        images=[row["images"]],
                        return_tensors="pt",
                    )
                    length = inputs["input_ids"].shape[-1]
                else:
                    length = len(
                        processor.apply_chat_template(
                            row["prompt"], tokenize=True, add_generation_prompt=True
                        )
                    )
                if length > args.max_length:
                    raise ValueError(f"GRPO prompt {index} exceeds its token budget")
        run_manifest["supervision"] = supervision
        run_manifest["status"] = "preflight_complete"
        save_manifest()
        if args.preflight_only:
            if world_size > 1:
                distributed.wait_for_everyone()
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            return
    except Exception as exc:
        run_manifest["status"] = "preflight_failed"
        run_manifest["error"] = str(exc)
        save_manifest()
        raise
    model_kwargs = {
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16" if common["bf16"] else "float32",
    }
    if args.revision:
        model_kwargs["revision"] = args.revision
    common["model_init_kwargs"] = model_kwargs
    if args.method == "sft":
        config = SFTConfig(
            **common,
            max_length=None if visual else args.max_length,
            completion_only_loss=True,
            packing=False,
        )
        trainer = VerifiedSFTTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            peft_config=peft,
            processing_class=processor,
        )
    else:
        from .rewards import CADReward

        reward = CADReward(
            dataset_root / "episodes",
            Path(args.output) / "cad-rollouts",
            executable=args.freecad,
            timeout=args.cad_timeout,
            workers=args.cad_workers,
        )
        config = GRPOConfig(
            **common,
            num_generations=args.num_generations,
            max_prompt_length=args.max_length,
            max_completion_length=args.max_completion_length,
            remove_unused_columns=False,
            use_vllm=False,
            beta=args.kl_beta,
        )
        trainer = VerifiedGRPOTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            reward_funcs=reward,
            processing_class=processor,
            peft_config=peft,
        )
    if args.freeze_vision:
        for name, parameter in trainer.model.named_parameters():
            if "visual" in name or "vision_tower" in name:
                parameter.requires_grad = False
    from .guards import make_numerical_guard

    guard = make_numerical_guard(args.require_learning_signal)
    trainer.add_callback(guard)
    trainer.model.config.use_cache = False
    run_manifest["trainable_parameters"] = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    run_manifest["vision_trainable_parameters"] = sum(
        p.numel()
        for name, p in trainer.model.named_parameters()
        if p.requires_grad and ("visual" in name or "vision_tower" in name)
    )
    if run_manifest["trainable_parameters"] == 0:
        raise ValueError("No trainable parameters")
    run_manifest["total_parameters"] = sum(
        p.numel() for p in trainer.model.parameters()
    )
    run_manifest["status"] = "running"
    save_manifest()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.output)
        if rank == 0 and trainer.processing_class is not None:
            trainer.processing_class.save_pretrained(args.output)
        run_manifest["status"] = "optimization_complete"
        run_manifest["train_metrics"] = train_result.metrics
        run_manifest["gradient_signal"] = guard.summary()
        run_manifest["resume_verification"] = getattr(
            trainer, "cad_resume_report", None
        )
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = str(exc)
        raise
    finally:
        hardware = {"rank": rank, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            hardware.update(
                device=torch.cuda.get_device_name(),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            )
        (out / f"rank-{rank}-memory.json").write_text(json.dumps(hardware, indent=2))
        save_manifest()
    if world_size > 1:
        distributed.wait_for_everyone()
    run_manifest["status"] = "complete"
    run_manifest["global_completion_verified"] = True
    save_manifest()
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
