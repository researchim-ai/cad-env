"""Merge a local SFT LoRA adapter into a separately reloadable policy checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--state-only", action="store_true")
    args = parser.parse_args()
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    from .policy import checkpoint_identity

    adapter = Path(args.adapter).resolve()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new merged checkpoint directory")
    config = PeftConfig.from_pretrained(adapter)
    base = config.base_model_name_or_path
    identity = checkpoint_identity(base, args.revision)
    manifest_path = adapter / "cad-training-run.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest.get("model", {})
        if recorded.get("revision") and recorded["revision"] != args.revision:
            raise ValueError("Base model revision differs from the training run")
    cls = AutoModelForCausalLM if args.state_only else AutoModelForImageTextToText
    kwargs = {"revision": args.revision} if args.revision else {}
    model = cls.from_pretrained(
        base, torch_dtype=torch.float32, attn_implementation="sdpa", **kwargs
    )
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    proc = (
        AutoTokenizer.from_pretrained(adapter)
        if args.state_only
        else AutoProcessor.from_pretrained(adapter)
    )
    proc.save_pretrained(out)
    if manifest_path.is_file():
        import shutil

        shutil.copyfile(manifest_path, out / "cad-training-run.json")
    (out / "merge-manifest.json").write_text(
        json.dumps(
            {
                "adapter": str(adapter),
                "base": identity,
                "dtype": "float32",
                "note": "Reload under the desired training dtype; merging can need substantial host RAM.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
