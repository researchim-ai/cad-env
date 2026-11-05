"""V6: paired mm/inch requests, independent value and composition holdouts."""

import argparse
import json
from pathlib import Path

from .visual_tasks import canonical_digest, make_visual_group


def build_tasks(groups=128, ood_groups=16, start_seed=2000000):
    if groups < 4 or groups % 4 or ood_groups < 1 or start_seed < 2000000:
        raise ValueError(
            "Fresh seeds, positive OOD groups and a multiple of four IID groups required"
        )
    tasks = []
    for offset in range(groups + 2 * ood_groups):
        value_ood = offset >= groups + ood_groups
        composition = groups <= offset < groups + ood_groups
        count = 5 + offset % 2 if composition else 1 + offset % 4
        inches = (
            [0.4375, 0.5625][offset % 2]
            if value_ood
            else [0.375, 0.5, 0.625][offset % 3]
        )
        diameter = round(25.4 * inches, 6)
        tier = "unit_ood" if value_ood else "compositional" if composition else "iid"
        for labels in [False, True]:
            for unit in ["mm", "inches"]:
                variants = make_visual_group(
                    start_seed + offset,
                    selected_count=count,
                    force_split="test" if tier != "iid" else None,
                    tier=tier,
                    target_diameter=diameter,
                    public_diameter=f"{inches:g} inches"
                    if unit == "inches"
                    else f"{diameter:g} mm",
                    shuffle_edit_order=True,
                    entity_labels=labels,
                )
                ids = {
                    t["task_id"]: "v-"
                    + canonical_digest([t["task_id"], "v6", unit])[:24]
                    for t in variants
                }
                for task in variants:
                    task["task_id"] = ids[task["task_id"]]
                    task["condition_group"] += (
                        ("-labels" if labels else "-unlabelled") + "-" + unit
                    )
                    task["counterfactual_goal_images"] = [
                        f"episodes/{ids[Path(p).parent.name]}/goal.png"
                        for p in task["counterfactual_goal_images"]
                    ]
                    task["curriculum_version"] = "visual-grounding-v6-units"
                tasks.extend(variants)
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task identities")
    splits = {}
    for task in tasks:
        splits.setdefault(task["split_group"], set()).add(task["split"])
    if any(len(v) != 1 for v in splits.values()):
        raise ValueError("Physical split leakage")
    return tasks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--groups", type=int, default=128)
    p.add_argument("--ood-groups", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    from .factory import generate_dataset
    from .visual_audit import audit_visual

    tasks = build_tasks(args.groups, args.ood_groups)
    summary = generate_dataset(
        args.output,
        seeds=args.groups,
        start_seed=2000000,
        workers=args.workers,
        preferences=False,
        families=("visual_grounding",),
        modes=("edit",),
        task_specs=tasks,
    )
    summary.update(
        benchmark="visual-grounding-v6-units",
        supported_training_exports=["train.feedback.jsonl"],
        requested_seed_range=[2000000, 2000000 + args.groups + 2 * args.ood_groups],
        unit_training_inches=[0.375, 0.5, 0.625],
        unit_holdout_inches=[0.4375, 0.5625],
        note="Paired mm/inches and labelled/unlabelled conditions share physical splits. Composition and value holdouts are separate strata; TEST excluded from training.",
    )
    root = Path(args.output)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_visual(root)
    (root / "visual-scale-audit.json").write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "accepted": summary["accepted"],
                "rejected": summary["rejected"],
                "audit": audit["ok"],
            }
        )
    )
    if not audit["ok"] or summary["accepted"] != len(tasks):
        raise SystemExit("Incomplete or invalid V6 corpus")


if __name__ == "__main__":
    main()
