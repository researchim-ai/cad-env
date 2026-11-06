"""Local CUDA operator checks, separate from full-model numerical behavior."""

import argparse
import copy
import json
from pathlib import Path


def run():
    import torch
    import torch.nn.functional as F
    from causal_conv1d import causal_conv1d_fn
    from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "causal": {},
        "rms": [],
        "scope": "Local forward/input-gradient checks; not end-to-end equality or quality evidence",
    }

    def error(a, b):
        return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-8))

    for dtype in [torch.float32, torch.bfloat16]:
        x = torch.randn(2, 256, 513, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(256, 4, device="cuda", dtype=dtype, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        ww = w.detach().clone().requires_grad_()
        reference = F.silu(
            F.conv1d(x, w.unsqueeze(1), padding=3, groups=256)[..., :513]
        )
        actual = causal_conv1d_fn(xx, ww, activation="silu")
        coefficients = torch.randn_like(reference)
        (reference.float() * coefficients.float()).sum().backward()
        (actual.float() * coefficients.float()).sum().backward()
        values = {
            "forward_l2": error(reference, actual),
            "input_grad_l2": error(x.grad, xx.grad),
            "weight_grad_l2": error(w.grad, ww.grad),
        }
        report["causal"][str(dtype)] = values
        values["passed"] = max(values.values()) <= (
            0.002 if dtype == torch.float32 else 0.03
        )
    root = Path("artifacts/visual-followup-500-fast-v1/merged")
    shards = list(root.glob("*.safetensors"))
    weights = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            names = [
                n
                for n in f.keys()  # noqa: SIM118 -- safe_open is not a dict/iterable
                if (
                    ".layers.0.input_layernorm.weight" in n
                    or ".layers.3.self_attn.q_norm.weight" in n
                    or ".language_model.norm.weight" in n
                )
            ]
            for name in names:
                weights[name] = f.get_tensor(name)
    if not weights:
        raise ValueError("No actual pretrained norm weights found")
    for name, weight in weights.items():
        for dtype in [torch.float32, torch.bfloat16]:
            a = Qwen3_5RMSNorm(weight.numel()).to(device="cuda", dtype=dtype)
            a.weight.data.copy_(weight.to(device="cuda", dtype=dtype))
            a.requires_grad_(False)
            b = copy.deepcopy(a)
            _patch_rms_norm_module(b, offset=1.0, casting_mode="gemma", in_place=False)
            x = torch.randn(
                2, 31, weight.numel(), device="cuda", dtype=dtype, requires_grad=True
            )
            xx = x.detach().clone().requires_grad_()
            y = a(x)
            yy = b(xx)
            c = torch.randn_like(y)
            (y.float() * c.float()).sum().backward()
            (yy.float() * c.float()).sum().backward()
            values = {
                "weight": name,
                "dtype": str(dtype),
                "forward_l2": error(y, yy),
                "input_grad_l2": error(x.grad, xx.grad),
            }
            report["rms"].append(values)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
