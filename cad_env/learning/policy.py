"""Local HF policy with identical public prompt contract for evaluation and deployment."""

import hashlib
from pathlib import Path


def checkpoint_identity(model, revision=None):
    path = Path(model)
    if not path.is_dir():
        if not revision or len(revision) != 40:
            raise ValueError(
                "Remote model must be pinned to an immutable 40-character revision"
            )
        return {"model": model, "revision": revision}
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and (
            file.suffix in (".json", ".safetensors")
            or file.name in ("merges.txt", "vocab.txt")
        ):
            h = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            hashes[str(file.relative_to(path))] = h.hexdigest()
    if not hashes:
        raise ValueError("Checkpoint directory contains no model artifacts")
    return {"local_path": str(path.resolve()), "files": hashes}


class HFPolicy:
    def __init__(
        self,
        model,
        *,
        revision=None,
        visual=True,
        max_new_tokens=256,
        temperature=0.0,
        max_context=4096,
        min_pixels=56 * 56,
        max_pixels=256 * 256,
        chat_template_kwargs=None,
    ):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        self.visual = visual
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_context = max_context
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.identity = checkpoint_identity(model, revision)
        if (Path(model) / "adapter_config.json").is_file():
            raise ValueError(
                "Merge the adapter into a frozen standalone checkpoint before evaluation"
            )
        kwargs = {"revision": revision} if revision else {}
        self.processor = (
            AutoProcessor.from_pretrained(
                model, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
            )
            if visual
            else AutoTokenizer.from_pretrained(model, **kwargs)
        )
        cls = AutoModelForImageTextToText if visual else AutoModelForCausalLM
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = cls.from_pretrained(
            model, torch_dtype=dtype, attn_implementation="sdpa", **kwargs
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.config.use_cache = True
        self.tokenizer = self.processor.tokenizer if visual else self.processor
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, messages, image_paths=()):
        import torch
        from PIL import Image

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        if self.visual:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            inputs = self.processor(
                text=[text], images=images or None, return_tensors="pt", padding=True
            )
        else:
            inputs = self.processor(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] + self.max_new_tokens > self.max_context:
            raise ValueError(
                "Policy context exceeds the fixed evaluation budget; no silent truncation is allowed"
            )
        import inspect

        if "token_type_ids" not in inspect.signature(self.model.forward).parameters:
            inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            options["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
