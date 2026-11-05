"""Fresh multi-edit visual curriculum; old frozen tests remain untouched."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .visual_tasks import make_visual_group


def build_tasks(groups=512, ood_groups=32, start_seed=1000000):
    if groups < 4 or groups % 4 or ood_groups < 1 or start_seed < 1000000:
        raise ValueError("Use a fresh seed range and a multiple of four IID groups")
    tasks = []
    for offset in range(groups + ood_groups):
        ood = offset >= groups
        count = (5 + (offset - groups) % 2) if ood else 1 + offset % 4
        # Units are learned in TRAIN as well; novel unit magnitudes must be a
        # separate holdout. All desired diameters exceed initial 6/8 mm holes.
        units = offset % 3 == 0
        diameter = 15.875 if units else [10, 12, 14, 16][(offset // 4) % 4]
        for labels in [False, True]:
            variants = make_visual_group(
                start_seed + offset,
                selected_count=count,
                force_split="test" if ood else None,
                tier="compositional" if ood else "iid",
                target_diameter=diameter,
                public_diameter="0.625 inches" if units else None,
                shuffle_edit_order=True,
                entity_labels=labels,
            )
            for task in variants:
                task["condition_group"] += "-labels" if labels else "-unlabelled"
                task["curriculum_version"] = "visual-grounding-v5-multiedit"
            tasks.extend(variants)
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tasks")
    lineage = {}
    for task in tasks:
        lineage.setdefault(task["split_group"], set()).add(task["split"])
    if any(len(splits) != 1 for splits in lineage.values()):
        raise ValueError("Physical lineage split leakage")
    return tasks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--groups", type=int, default=512)
    p.add_argument("--ood-groups", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    from .factory import generate_dataset
    from .visual_audit import audit_visual

    tasks = build_tasks(args.groups, args.ood_groups)
    summary = generate_dataset(
        args.output,
        seeds=args.groups,
        workers=args.workers,
        preferences=False,
        families=("visual_grounding",),
        modes=("edit",),
        task_specs=tasks,
    )
    summary.update(
        benchmark="visual-grounding-v5-multiedit",
        teacher_order="shuffled",
        supported_training_exports=["train.feedback.jsonl"],
        selection_counts=dict(Counter(len(t["goal_spec"]["selected"]) for t in tasks)),
        interpretation="TRAIN 1-4 edits, labelled and unlabelled counterfactual goals, mm/inches; fresh physical splits; TEST 5-6 edits. Not directly comparable to old V3/V4 test after training on this curriculum.",
    )
    root = Path(args.output)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_visual(root)
    (root / "visual-scale-audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
