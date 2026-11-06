"""Fresh validation parents for the V6 protocol; never exported into training."""

import argparse
import json
from pathlib import Path

from .visual_units import build_tasks


def validation_tasks():
    tasks = build_tasks(groups=1024, ood_groups=1, start_seed=3000000)
    groups = {}
    for task in tasks:
        if task["split"] == "validation":
            groups.setdefault(task["split_group"], []).append(task)
    selected = []
    counts = {n: 0 for n in range(1, 5)}
    for variants in groups.values():
        count = len(variants[0]["oracle_actions"]) - 1
        if count in counts and counts[count] < 10:
            selected.extend(variants)
            counts[count] += 1
    if list(counts.values()) != [10] * 4:
        raise ValueError(f"Insufficient balanced validation parents: {counts}")
    return selected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from .factory import generate_dataset
    from .visual_audit import audit_visual

    tasks = validation_tasks()
    summary = generate_dataset(
        args.output,
        workers=4,
        preferences=False,
        families=("visual_grounding",),
        modes=("edit",),
        task_specs=tasks,
    )
    audit = audit_visual(args.output)
    report = {
        "parents": 40,
        "tasks": len(tasks),
        "accepted": summary["accepted"],
        "audit": audit,
        "scope": "Fresh synthetic IID validation, 10 parents per edit count; no TEST or TRAIN rows",
    }
    Path(args.output, "holdout-audit.json").write_text(json.dumps(report, indent=2))
    if summary["accepted"] != len(tasks) or not audit["ok"]:
        raise RuntimeError("Fresh validation failed native/visual audit")


if __name__ == "__main__":
    main()
