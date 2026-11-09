"""Audit and process document inputs with the actual pinned VLM processor.

CPU preflight only: this is not a native task-success or GPU memory benchmark.
"""

import argparse
import json
from pathlib import Path

from .dataset import audit_technical_dataset


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("dataset", "model", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    root = Path(a.dataset)
    audit = audit_technical_dataset(root)
    if not audit["ok"]:
        raise ValueError(str(audit["errors"][:3]))
    from transformers import AutoProcessor

    from cad_env.learning.extreme_benchmark import context_preflight

    processor = AutoProcessor.from_pretrained(
        a.model, min_pixels=56 * 56, max_pixels=327680
    )
    profile = context_preflight(root, processor, 32768)
    result = {
        "status": "preflight_complete",
        "audit": audit,
        "context": profile,
        "model": a.model,
        "max_context": 32768,
        "max_pixels_per_image": 327680,
        "scope": "CPU document-input preflight only; no claim about GPU memory or task success",
    }
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "status": result["status"],
                "maximum_prompt_tokens": profile["maximum_prompt_tokens"],
            }
        )
    )


if __name__ == "__main__":
    main()
