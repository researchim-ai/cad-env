"""Isolated, pinned SFT path for Qwen3.5; legacy TRL 0.24 stays unchanged."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from pathlib import Path


def nonthinking_template(template):
    if not isinstance(template, str) or not template:
        raise ValueError("A single explicit model chat template is required")
    return "{%- set enable_thinking = false -%}" + template


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--extra-dataset", action="append", default=[])
    p.add_argument(
        "--source-row-budgets",
        type=int,
        nargs="+",
        help="Positive row budget in dataset/extra-dataset order",
    )
    p.add_argument(
        "--rows-per-source",
        type=int,
        help="Explicit equal row budget per unique dataset, cycling small sources",
    )
    p.add_argument(
        "--vision-lora",
        action="store_true",
        help="Adapt the last eight vision attention blocks and visual merger",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--numeric-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--max-train-rows",
        type=int,
        help="Train-only deterministic subset for GPU smoke runs",
    )
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-images", type=int, default=1)
    p.add_argument("--max-pixels", type=int, default=65536)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume-from-checkpoint")
    args = p.parse_args()
    data_roots = [Path(args.dataset).resolve()] + [
        Path(path).resolve() for path in args.extra_dataset
    ]
    if len(set(data_roots)) != len(data_roots):
        p.error(
            "Specify each dataset once; repeated roots do not define a sampling weight"
        )
    if args.rows_per_source is not None and args.rows_per_source < 1:
        p.error("rows-per-source must be positive")
    if args.source_row_budgets is not None and (
        args.rows_per_source is not None
        or len(args.source_row_budgets) != len(data_roots)
        or min(args.source_row_budgets) < 1
    ):
        p.error(
            "Use one positive source-row-budget per dataset, without rows-per-source"
        )
    if args.max_train_rows is not None and args.max_train_rows < 1:
        p.error("max-train-rows must be positive")
    if not math.isfinite(args.numeric_loss_weight) or args.numeric_loss_weight < 1:
        p.error("Numeric loss weight must be finite and at least one")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This measured modern profile supports one GPU")
    if (
        min(
            args.max_steps,
            args.save_steps,
            args.gradient_accumulation,
            args.max_length,
            args.batch_size,
        )
        < 1
    ):
        raise ValueError("Invalid training budget")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "accelerate",
            "datasets",
            "tokenizers",
            "safetensors",
        )
    }
    expected = {"transformers": "5.22.1", "trl": "1.17.1", "peft": "0.21.2"}
    if any(versions[k] != v for k, v in expected.items()):
        raise ValueError("Use the isolated pinned Qwen3.5 environment")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    from .checkpoints import StrictResumeMixin
    from .feedback_normalization import normalize_feedback_prompt
    from .guards import make_numerical_guard

    class VerifiedModernSFTTrainer(StrictResumeMixin, SFTTrainer):
        pass

    from .policy import checkpoint_identity
    from .train import check_training_source, check_vision_supervision, load_export

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires a BF16-capable CUDA GPU")
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new run output")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset).resolve()
    source = root / "exports/train.feedback.jsonl"
    data_audit = check_training_source(root, source)
    manifest = {
        "status": "preflight",
        "arguments": vars(args),
        "versions": versions,
        "model": checkpoint_identity(args.model),
        "action_contract": "single_action",
        "chat_template_kwargs": {"enable_thinking": False},
        "dataset_digest": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
        "export_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(0),
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "data_audit": data_audit,
    }

    def persist():
        temporary = out / "cad-training-run.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(out / "cad-training-run.json")

    persist()
    for folder in ("agent", "learning", "multicad"):
        for file in (Path(__file__).resolve().parents[1] / folder).glob("*.py"):
            target = out / "source_snapshot" / folder / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    manifest["source_sha256"] = {
        str(f.relative_to(out / "source_snapshot")): hashlib.sha256(
            f.read_bytes()
        ).hexdigest()
        for f in sorted((out / "source_snapshot").rglob("*.py"))
    }
    (out / "environment.json").write_text(
        json.dumps(
            {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            },
            indent=2,
        )
    )
    try:
        set_seed(42)
        processor = AutoProcessor.from_pretrained(
            args.model, min_pixels=56 * 56, max_pixels=args.max_pixels
        )
        processor.chat_template = nonthinking_template(processor.chat_template)
        datasets = [load_export(source, dataset_root=root)]
        manifest["additional_datasets"] = []
        group_splits = {}
        for data_root in [root] + [Path(p).resolve() for p in args.extra_dataset]:
            if data_root != root:
                extra_source = data_root / "exports/train.feedback.jsonl"
                extra_audit = check_training_source(data_root, extra_source)
                datasets.append(load_export(extra_source, dataset_root=data_root))
                manifest["additional_datasets"].append(
                    {
                        "root": str(data_root),
                        "data_audit": extra_audit,
                        "dataset_digest": hashlib.sha256(
                            (data_root / "summary.json").read_bytes()
                        ).hexdigest(),
                        "export_sha256": hashlib.sha256(
                            extra_source.read_bytes()
                        ).hexdigest(),
                    }
                )
            for line in (data_root / "manifest.jsonl").read_text().splitlines():
                entry = json.loads(line)
                if entry["accepted"]:
                    group_splits.setdefault(entry["split_group"], set()).add(
                        entry["split"]
                    )
        if any(len(splits) != 1 for splits in group_splits.values()):
            raise ValueError("Cross-dataset lineage split leakage")
        from .mixing import mix_sources

        manifest["source_mixing"] = {
            "rows_per_source": args.rows_per_source,
            "source_row_budgets": args.source_row_budgets,
            "source_rows": [len(d) for d in datasets],
            "seed": 42,
        }
        dataset = mix_sources(
            datasets, args.source_row_budgets or args.rows_per_source, seed=42
        )
        if args.max_train_rows is not None:
            dataset = dataset.shuffle(seed=42).select(
                range(min(len(dataset), args.max_train_rows))
            )
        manifest["selected_training_rows"] = len(dataset)
        dataset = dataset.map(
            lambda row: {"prompt": normalize_feedback_prompt(row["prompt"])}
        )
        manifest["prompt_normalization"] = (
            "single_action_v3: single action guide and canonical public teacher responses"
        )
        manifest["supervision"] = check_vision_supervision(
            dataset,
            processor,
            args.max_length,
            max_images=args.max_images,
            image_token_ids=[AutoConfig.from_pretrained(args.model).image_token_id],
            verify_template_boundary=True,
        )
        processor.save_pretrained(out)
        persist()
        if args.preflight_only:
            manifest["status"] = "preflight_complete"
            persist()
            return
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$"
        if args.vision_lora:
            targets = (
                "(?:"
                + targets
                + r"|.*visual\.blocks\.(1[6-9]|2[0-3])\.attn\.(qkv|proj)$"
                + r"|.*visual\.merger\.linear_fc[12]$)"
            )
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, config)
        for name, parameter in model.named_parameters():
            if "visual" in name or "vision_tower" in name:
                parameter.requires_grad = args.vision_lora and "lora_" in name
        manifest["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        manifest["vision_trainable_parameters"] = sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and ("visual" in n or "vision_tower" in n)
        )
        if (
            not manifest["trainable_parameters"]
            or bool(manifest["vision_trainable_parameters"]) != args.vision_lora
        ):
            raise RuntimeError("Unexpected trainable parameter set")
        model.config.use_cache = False
        guard = make_numerical_guard(require_signal=True)

        class VisionGradientGuard(TrainerCallback):
            def __init__(self):
                self.nonzero_steps = 0
                self.max_norm = 0.0

            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                if not args_vision:
                    return
                norms = [
                    p.grad.detach().float().norm()
                    for n, p in model.named_parameters()
                    if "visual" in n and p.requires_grad and p.grad is not None
                ]
                norm = torch.stack(norms).norm().item() if norms else 0.0
                if not math.isfinite(norm):
                    raise FloatingPointError("Non-finite visual adapter gradient")
                self.nonzero_steps += int(norm > 0)
                self.max_norm = max(self.max_norm, norm)

            def on_train_end(self, args, state, control, **kwargs):
                if args_vision and not self.nonzero_steps:
                    raise RuntimeError("Vision adapters received no gradient")

        args_vision = args.vision_lora
        vision_guard = VisionGradientGuard()
        numeric_loss = None
        if args.numeric_loss_weight != 1:
            from .numeric_loss import NumericCompletionLoss

            numeric_loss = NumericCompletionLoss(
                processor.tokenizer, args.numeric_loss_weight
            )
        trainer = VerifiedModernSFTTrainer(
            model=model,
            processing_class=processor,
            train_dataset=dataset,
            callbacks=[guard, vision_guard],
            compute_loss_func=numeric_loss,
            args=SFTConfig(
                output_dir=str(out),
                max_steps=args.max_steps,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.learning_rate,
                bf16=True,
                fp16=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=1,
                logging_nan_inf_filter=False,
                report_to="none",
                save_steps=args.save_steps,
                save_total_limit=3,
                save_strategy="steps",
                max_length=None,
                completion_only_loss=True,
                packing=False,
                seed=42,
                dataloader_num_workers=0,
            ),
        )
        manifest["status"] = "running"
        persist()
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(out))
        processor.save_pretrained(out)
        manifest.update(
            status="complete",
            train_metrics=result.metrics,
            gradient_signal=guard.summary(),
            visual_gradient_signal={
                "nonzero_steps": vision_guard.nonzero_steps,
                "max_grad_norm": vision_guard.max_norm,
            },
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            numeric_loss=numeric_loss.summary() if numeric_loss else None,
            global_completion_verified=True,
            resume_verification=getattr(trainer, "cad_resume_report", None),
        )
        persist()
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
