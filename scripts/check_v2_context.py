"""CPU-only token/image budget check using the pinned model processor, without weights."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-images", type=int, default=22)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=56 * 56, max_pixels=327680, local_files_only=True
    )
    root = args.dataset.absolute()
    cache = {}
    records = []
    for export in sorted((root / "exports").glob("*.feedback.jsonl")):
        for line in export.read_text().splitlines():
            row = json.loads(line)
            key = tuple(row["images"])
            if not 1 <= len(key) <= args.max_images:
                raise ValueError("Image count outside budget")
            if key not in cache:
                images = []
                for relative in key:
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root.resolve()):
                        raise ValueError("Unsafe image path")
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                probe = [{"role": "user", "content": [{"type": "image"} for _ in key]}]
                text = processor.apply_chat_template(
                    probe,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                expanded = processor(text=[text], images=images, return_tensors="pt")[
                    "input_ids"
                ].shape[-1]
                base = len(processor.tokenizer(text)["input_ids"])
                cache[key] = {"delta": expanded - base, "images": images, "worst": None}
            text = processor.apply_chat_template(
                row["prompt"] + row["completion"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            size = len(processor.tokenizer(text)["input_ids"]) + cache[key]["delta"]
            record = {
                "task_id": row["task_id"],
                "step_id": row["step_id"],
                "standard": row.get("standard"),
                "tokens": size,
                "images": len(key),
                "export": export.name,
            }
            records.append(record)
            if cache[key]["worst"] is None or size > cache[key]["worst"][0]:
                cache[key]["worst"] = (size, text)
    if not records:
        raise ValueError("No examples")
    # Verify the cached expansion estimate with a full real processor call on
    # the longest example for every distinct set of images.
    for item in cache.values():
        expected, text = item["worst"]
        actual = processor(text=[text], images=item["images"], return_tensors="pt")[
            "input_ids"
        ].shape[-1]
        if expected != actual:
            raise ValueError("Processor expansion estimate mismatch")
    oversized = [r for r in records if r["tokens"] > args.max_length]
    report = {
        "ok": not oversized,
        "rows": len(records),
        "distinct_image_sets": len(cache),
        "max_tokens": max(r["tokens"] for r in records),
        "max_images": max(r["images"] for r in records),
        "context_limit": args.max_length,
        "oversized": oversized,
        "worst_examples": sorted(records, key=lambda r: r["tokens"], reverse=True)[:10],
        "scope": "CPU processor/token budget only; not model inference, visual grounding, completion-mask validation or learning evidence.",
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("oversized", "worst_examples")
            },
            indent=2,
        )
    )
    if oversized:
        raise ValueError("Training examples exceed context; no truncation permitted")


if __name__ == "__main__":
    main()
