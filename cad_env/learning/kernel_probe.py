"""Small real Qwen3.5 VLM backward correctness probe, not a speed benchmark."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace


def run(device="cpu", liger=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from trl.trainer.utils import add_fused_lm_head

    from .fused_numeric import apply_liger_layers, fused_numeric_loss
    from .numeric_loss import NumericCompletionLoss

    torch.manual_seed(42)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=7,
        video_token_id=10,
        vision_start_token_id=8,
        vision_end_token_id=9,
    )
    config._attn_implementation = "sdpa"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = Qwen3_5ForConditionalGeneration(config).to(device=device, dtype=dtype)
    reference = copy.deepcopy(model)
    if liger:
        apply_liger_layers(model)
    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "qkv", "linear_fc1", "linear_fc2"],
    )
    torch.manual_seed(17)
    reference = get_peft_model(reference, lora)
    torch.manual_seed(17)
    model = get_peft_model(model, lora)
    inputs = {
        "input_ids": torch.tensor([[8, 7, 9, 3, 1, 2]], device=device),
        "attention_mask": torch.ones(1, 6, dtype=torch.long, device=device),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0]], device=device),
        "pixel_values": torch.randn(4, 1536, device=device, dtype=dtype),
        "image_grid_thw": torch.tensor([[1, 2, 2]], device=device),
        "use_cache": False,
    }
    labels = torch.tensor([[-100, -100, -100, 3, 1, 2]], device=device)
    tokenizer = SimpleNamespace(get_vocab=lambda: {"1": 1, "2": 2})
    original = NumericCompletionLoss(tokenizer)
    optimized = NumericCompletionLoss(tokenizer)
    expected = original(reference(**inputs), labels, 9)
    expected.backward()
    add_fused_lm_head(
        model.get_base_model(), outputs=("log_probs", "entropy", "is_top1")
    )
    outputs = model(**inputs, labels=labels, fused_lm_head=True)
    actual = fused_numeric_loss(outputs, labels, optimized, 9)
    actual.backward()
    torch.testing.assert_close(
        actual, expected, rtol=0.01 if device != "cpu" else 1e-5, atol=1e-5
    )
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            before = dict(reference.named_parameters())[name].grad
            if parameter.grad is None or before is None:
                raise AssertionError("Missing adapter gradient: " + name)
            delta = (parameter.grad.float() - before.float()).norm()
            relative = float(delta / before.float().norm().clamp_min(1e-8))
            if relative > (0.06 if device != "cpu" else 1e-4):
                raise AssertionError(f"Gradient mismatch {name}: {relative}")
            gradients[name] = {
                "relative_error": relative,
                "norm": float(parameter.grad.float().norm()),
            }
    vision = [g for n, g in gradients.items() if "visual" in n]
    assert vision and any(g["norm"] > 0 for g in vision)
    return {
        "device": device,
        "dtype": str(dtype),
        "liger": liger,
        "loss_reference": float(expected.detach()),
        "loss_optimized": float(actual.detach()),
        "adapter_gradients": gradients,
        "numeric_statistics_equal": original.summary() == optimized.summary(),
        "scope": "Tiny random Qwen3.5 VLM, real image input and visual/language LoRA backward; not 4B throughput or CAD quality",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--liger", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = run(args.device, args.liger)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "adapter_gradients"}))


if __name__ == "__main__":
    main()
