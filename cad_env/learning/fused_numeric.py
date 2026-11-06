"""Single-forward numeric SFT loss for the pinned TRL fused LM head.

Opt-in only: the running baseline and its frozen source files stay unchanged.
"""


def fused_numeric_loss(outputs, labels, numeric_loss, num_items_in_batch=None):
    import torch

    targets = labels[..., 1:]
    valid = targets != -100
    if outputs.log_probs.shape != targets.shape or not torch.equal(
        outputs.label_mask, valid
    ):
        raise ValueError("Fused next-token alignment or completion mask differs")
    if not valid.any():
        raise ValueError("No supervised next-token targets")
    ids = torch.tensor(numeric_loss.numeric_ids, device=targets.device)
    numeric = torch.isin(targets, ids) & valid
    weights = 1 + numeric.float() * (numeric_loss.weight - 1)
    denominator = num_items_in_batch if num_items_in_batch is not None else valid.sum()
    with torch.no_grad():
        numeric_loss.numeric_tokens += int(numeric.sum())
        numeric_loss.numeric_correct += int((outputs.is_top1 & numeric).sum())
        numeric_loss.supervised_tokens += int(valid.sum())
    return (-outputs.log_probs[valid].float() * weights[valid]).sum() / denominator


def trainer_class(base):
    """Create an opt-in subclass; never globally monkey-patch a running process."""

    class FusedNumericSFTTrainer(base):
        def __init__(self, *args, compute_loss_func=None, **kwargs):
            from .numeric_loss import NumericCompletionLoss

            if not isinstance(compute_loss_func, NumericCompletionLoss):
                raise TypeError("This path requires NumericCompletionLoss")
            self.cad_numeric_loss = compute_loss_func
            super().__init__(*args, compute_loss_func=None, **kwargs)
            if self.accelerator.num_processes != 1 or self.aux_loss_enabled:
                raise ValueError("Validated profile is single-GPU dense SFT only")
            # The loss divides by the entire accumulation-window target count.
            self.model_accepts_loss_kwargs = True

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            if return_outputs or "shift_labels" in inputs:
                raise ValueError(
                    "Training-only path; shifted labels/eval outputs require a separate validated path"
                )
            inputs = {**inputs, "use_cache": False, "fused_lm_head": True}
            outputs = model(**inputs)
            loss = fused_numeric_loss(
                outputs, inputs["labels"], self.cad_numeric_loss, num_items_in_batch
            )
            mode = "train" if model.training else "eval"
            valid = outputs.label_mask.sum()
            self._metrics[mode]["entropy"].append(
                (outputs.entropy.detach().sum() / valid).item()
            )
            self._metrics[mode]["mean_token_accuracy"].append(
                (outputs.is_top1.sum() / valid).item()
            )
            if model.training:
                self._total_train_tokens += inputs["attention_mask"].sum().item()
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            return loss

    return FusedNumericSFTTrainer


def apply_liger_layers(model, profile="all"):
    """Apply only the tested dense text RMSNorm/SwiGLU kernels, before PEFT."""
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    if model.config.model_type != "qwen3_5":
        raise ValueError("Qwen3.5 dense VLM required")
    if profile not in {"all", "rms", "swiglu"}:
        raise ValueError("Unknown Liger layer profile")
    apply_liger_kernel_to_qwen3_5(
        model=model,
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=profile in {"all", "rms"},
        swiglu=profile in {"all", "swiglu"},
    )


def checkpoint_linear_layers_only(model):
    """Keep large Gated DeltaNet activations checkpointed; retain full-attention activations."""
    base = model.get_base_model()
    original = base.gradient_checkpointing_enable

    def enable(*args, **kwargs):
        original(*args, **kwargs)
        disabled = 0
        for layer in base.model.language_model.layers:
            if layer.block_type == "full_attention":
                layer.gradient_checkpointing = False
                disabled += 1
        if not disabled:
            raise ValueError(
                "Hybrid policy requires Qwen3.5 full-attention decoder layers"
            )

    base.gradient_checkpointing_enable = enable
