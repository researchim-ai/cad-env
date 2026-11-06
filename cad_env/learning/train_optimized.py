"""Opt-in launcher for the single-forward numeric loss and optional Liger layers.

Uses the unchanged modern trainer's data audits, callbacks and manifest. Never
changes the running baseline. Only use after the correctness/throughput probes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--liger-layers", action="store_true")
    p.add_argument("--liger-profile", choices=["all", "rms", "swiglu"], default="all")
    p.add_argument("--output", required=True)
    p.add_argument("--memory-preflight", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--checkpoint-policy", choices=["all", "linear"], default="all")
    options, remaining = p.parse_known_args()
    if options.no_gradient_checkpointing and options.checkpoint_policy != "all":
        p.error("Choose checkpoint policy or disable checkpointing, not both")
    import trl

    from .fused_numeric import apply_liger_layers, trainer_class

    optimized = trainer_class(trl.SFTTrainer)
    if options.checkpoint_policy == "linear":
        from .fused_numeric import checkpoint_linear_layers_only

        class HybridCheckpointTrainer(optimized):
            def __init__(self, *args, **kwargs):
                checkpoint_linear_layers_only(kwargs["model"])
                super().__init__(*args, **kwargs)

        optimized = HybridCheckpointTrainer
    if options.no_gradient_checkpointing:

        class UncheckpointedTrainer(optimized):
            def __init__(self, *args, **kwargs):
                kwargs["args"].gradient_checkpointing = False
                super().__init__(*args, **kwargs)

        optimized = UncheckpointedTrainer
    if options.liger_layers:
        version = importlib.metadata.version("liger-kernel")
        if version != "0.8.5":
            raise ValueError("Only the isolated, probed Liger 0.8.5 profile is allowed")

        class LigerNumericTrainer(optimized):
            def __init__(self, *args, **kwargs):
                model = kwargs["model"]
                apply_liger_layers(model.get_base_model(), options.liger_profile)
                super().__init__(*args, **kwargs)

        optimized = LigerNumericTrainer
    if options.memory_preflight:
        import copy
        import time

        import torch

        previous = optimized

        class MemoryCheckedTrainer(previous):
            def __init__(self, *args, **kwargs):
                dataset = kwargs["train_dataset"]
                super().__init__(*args, **kwargs)
                report_path = Path(options.output, "cad-training-run.json")
                report = json.loads(report_path.read_text())
                size = self.args.per_device_train_batch_size
                indices = report["supervision"]["largest_row_indices"][:size]
                if len(indices) != size:
                    raise ValueError(
                        "Insufficient distinct rows for batch memory probe"
                    )
                batch = self._prepare_inputs(
                    self.data_collator([dataset[i] for i in indices])
                )
                rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                metrics, tokens = copy.deepcopy(self._metrics), self._total_train_tokens
                counters = (
                    self.cad_numeric_loss.numeric_tokens,
                    self.cad_numeric_loss.numeric_correct,
                    self.cad_numeric_loss.supervised_tokens,
                )
                if self.args.gradient_checkpointing:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                self.model.train()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.compute_loss(self.model, batch)
                loss.backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                memory_report = {
                    "rows": indices,
                    "batch_size": size,
                    "peak_allocated_gib": peak,
                    "elapsed_seconds": time.monotonic() - started,
                    "optimizer_steps": 0,
                    "scope": "Forward/backward on longest audited selected rows; optimizer moments not yet allocated, requires 2 GiB headroom",
                }
                Path(options.output, "memory-preflight.json").write_text(
                    json.dumps(memory_report, indent=2)
                )
                self.model.zero_grad(set_to_none=True)
                del loss, batch
                torch.cuda.empty_cache()
                torch.set_rng_state(rng)
                torch.cuda.set_rng_state(cuda_rng)
                self._metrics, self._total_train_tokens = metrics, tokens
                (
                    self.cad_numeric_loss.numeric_tokens,
                    self.cad_numeric_loss.numeric_correct,
                    self.cad_numeric_loss.supervised_tokens,
                ) = counters
                if (
                    peak
                    > torch.cuda.get_device_properties(0).total_memory / 1024**3 - 2
                ):
                    raise RuntimeError(
                        "Worst-row memory probe lacks optimizer/shape headroom; refusing training"
                    )

        optimized = MemoryCheckedTrainer
    # This substitution is confined to this opt-in process. The baseline module
    # imports SFTTrainer inside main, so all its ordinary audits remain active.
    trl.SFTTrainer = optimized
    sys.argv = [sys.argv[0], *remaining, "--output", options.output]
    from .train_modern import main as train

    train()
    from transformers.models.qwen3_5 import modeling_qwen3_5

    report = {
        "single_forward_numeric_loss": True,
        "liger_layers": options.liger_layers,
        "liger_profile": options.liger_profile if options.liger_layers else None,
        "liger_version": importlib.metadata.version("liger-kernel")
        if options.liger_layers
        else None,
        "fused_linear_cross_entropy": False,
        "rope_patch": False,
        "gradient_checkpointing": not options.no_gradient_checkpointing,
        "checkpoint_policy": options.checkpoint_policy,
        "causal_conv1d_active": modeling_qwen3_5.causal_conv1d_fn is not None,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("fused_numeric.py")]
        },
    }
    Path(options.output, "optimized-runtime.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
