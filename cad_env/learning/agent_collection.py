"""Stratified train-only, on-policy tool rollouts and immutable recovery bundles."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .audit import audit_dataset
from .bundles import bundle_supervision
from .factory import jsonl
from .rollouts import collect_group
from .workflow import grpo_readiness


def collect_agents(
    dataset, policy, output, *, prompts=20, num_generations=4, max_steps=16, seed=42
):
    root = Path(dataset).resolve()
    out = Path(output).resolve()
    if not audit_dataset(root)["ok"]:
        raise ValueError("Dataset audit failed")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new collection directory")
    out.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)
    for line in (root / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["accepted"] and row["split"] == "train":
            buckets[(row["family"], row["mode"])].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < prompts and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < prompts:
                selected.append(buckets[key].pop())
    if not selected:
        raise ValueError("No train tasks to collect")
    scores = []
    successes = []
    from transformers import set_seed

    set_seed(seed)
    for entry in selected:
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        task["max_steps"] = min(max_steps, task["max_steps"])
        group = out / "groups" / task["task_id"]
        collect_group(task, policy, group, num_generations=num_generations, visual=True)
        scores.extend(
            json.loads(line)
            for line in (group / "rollouts.jsonl").read_text().splitlines()
        )
        successes.extend(
            json.loads(line)
            for line in (group / "successful.steps.sft.jsonl").read_text().splitlines()
            if line.strip()
        )
        jsonl(out / "scores.jsonl", scores)
    digest = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    bundle = bundle_supervision(successes, out, parent_dataset_digest=digest)
    report = {
        "seed": seed,
        "prompts": len(selected),
        "generations_per_prompt": num_generations,
        "model_identity": getattr(policy, "identity", None),
        "parent_dataset_digest": digest,
        "supervision_rows": bundle["rows"],
        "successful_rollouts": sum(row["task_success"] for row in scores),
        "readiness": grpo_readiness(scores),
        "note": "Fresh train-only on-policy data. Validation/test tasks are never mined.",
    }
    (out / "collection.json").write_text(json.dumps(report, indent=2))
    return report
