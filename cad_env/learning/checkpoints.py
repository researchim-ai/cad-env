"""Strict resume for pinned Transformers, including VLM save-key conversions.

The standard Trainer in Transformers 4.57.0 calls load_state_dict directly,
which bypasses architecture-specific legacy-key conversions on dense VLM saves.
"""

from pathlib import Path


def restore_checkpoint_weights(model, checkpoint):
    import torch

    folder = Path(checkpoint)
    if hasattr(model, "peft_config"):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        file = folder / "adapter_model.safetensors"
        if not file.is_file():
            raise ValueError("A safetensors adapter checkpoint is required")
        state = load_file(file, device="cpu")
        result = set_peft_model_state_dict(model, state, adapter_name="default")
        if result.unexpected_keys:
            raise ValueError("Unexpected adapter keys: " + str(result.unexpected_keys))
        loaded = get_peft_model_state_dict(model, adapter_name="default")
        if set(loaded) != set(state):
            raise ValueError(
                "Adapter checkpoint key set differs from the active adapter"
            )
        for key, value in state.items():
            if not torch.equal(loaded[key].detach().cpu(), value.to(loaded[key].dtype)):
                raise ValueError("Adapter tensor failed round-trip: " + key)
        return {
            "kind": "adapter",
            "verified_tensors": len(state),
            "exact_roundtrip": True,
        }
    # Use the model's public loader so renamed VLM keys are converted correctly.
    restored = type(model).from_pretrained(
        folder,
        torch_dtype=model.dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    state = restored.state_dict()
    model.load_state_dict(state, strict=True)
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), state[key].detach().cpu()):
            raise ValueError("Dense tensor failed round-trip: " + key)
    count = len(state)
    del restored, state
    return {"kind": "dense", "verified_tensors": count, "exact_roundtrip": True}


class StrictResumeMixin:
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if getattr(self, "is_fsdp_enabled", False) or getattr(
            self, "is_deepspeed_enabled", False
        ):
            raise ValueError(
                "Strict resume currently supports ordinary/DDP models; FSDP/DeepSpeed requires its own tested path"
            )
        self.cad_resume_report = restore_checkpoint_weights(
            model if model is not None else self.model, resume_from_checkpoint
        )
