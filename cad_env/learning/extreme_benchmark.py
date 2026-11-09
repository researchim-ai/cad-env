"""Development calibration for large synthetic parts, with context preflight.

This benchmark does not certify industrial drawings or manufacturability.
Incomplete runs and context failures are infrastructure failures, never zero scores.
"""

import argparse
import json
import os
from pathlib import Path


def context_preflight(root, processor, maximum):
    from PIL import Image

    rows = {}
    for line in (root / "exports/validation.feedback.jsonl").read_text().splitlines():
        row = json.loads(line)
        # For each task choose the largest public prompt, including the native
        # final-state objects and recovery-history variants. Images are fixed.
        size = len(json.dumps(row["prompt"]))
        if row["task_id"] not in rows or size > rows[row["task_id"]][0]:
            rows[row["task_id"]] = (size, row)
    lengths = {}
    for task_id, (_, row) in rows.items():
        text = processor.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        images = []
        for relative in row["images"]:
            with Image.open(root / relative) as im:
                images.append(im.convert("RGB"))
        inputs = processor(text=[text], images=images, return_tensors="pt")
        count = inputs["input_ids"].shape[-1]
        lengths[task_id] = count
    if not lengths or max(lengths.values()) + 256 > maximum:
        raise ValueError(
            "Teacher context does not fit; do not interpret this as model failure: "
            + str(lengths)
        )
    return {
        "tasks": len(lengths),
        "maximum_prompt_tokens": max(lengths.values()),
        "per_task": lengths,
        "scope": "Largest serialized teacher prompt per validation task. Student-created states remain checked at every policy call.",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ("dataset", "model", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    a = parser.parse_args()
    if not a.preflight_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("GPU1 only")
    from transformers import AutoProcessor

    from .extreme_tasks import audit_extreme

    root, out = Path(a.dataset).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "preflight",
        "test_opened": False,
        "promoted": False,
        "benchmark_scope": "synthetic large compositions; standards conformance and manufacturing not assessed",
    }

    def persist():
        temp = out / "status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(out / "status.json")

    persist()
    batcher = None
    try:
        audit = audit_extreme(root)
        if not audit["ok"]:
            raise ValueError(
                "Native/semantic corpus audit failed: " + str(audit["errors"][:3])
            )
        maximum = 32768
        processor = AutoProcessor.from_pretrained(
            a.model, min_pixels=56 * 56, max_pixels=327680
        )
        status["context_preflight"] = context_preflight(root, processor, maximum)
        if a.preflight_only:
            status["status"] = "preflight_complete"
            return
        from .batch_policy import BatchedHFPolicy, PolicyBatcher
        from .parallel_protocol import evaluate_policy

        policy = BatchedHFPolicy(
            a.model,
            max_context=maximum,
            max_pixels=327680,
            temperature=0,
            max_new_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
        )
        batcher = PolicyBatcher(policy, batch_size=2)
        status["status"] = "evaluating"
        persist()
        report = evaluate_policy(
            root,
            lambda seed: batcher,
            out / "native",
            workers=4,
            max_steps=128,
            model_identity=policy.identity,
            policy_protocol={
                "max_context": maximum,
                "max_pixels": 327680,
                "max_new_tokens": 256,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        status.update(status="complete", metrics=report["metrics"])
        status["batch_stats"] = batcher.stats
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        if batcher is not None:
            batcher.close()
        persist()


if __name__ == "__main__":
    main()
