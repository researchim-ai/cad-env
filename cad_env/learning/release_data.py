"""Fresh, native-verified counterfactual clean/stop/repair curriculum.

Shared intermediate states across four different pixel goals prevent the state
itself from disclosing the correct target. Includes correct states that require
finish and incorrect states that require restoring protected diameters. All
parents keep natural hash splits; no TEST task geometry is generated or read.
"""

import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

from .visual_tasks import canonical_digest, make_visual_group


def known_parents(root=Path("artifacts")):
    # Episode execution trees contain no dataset manifests and can have millions
    # of files. Prune them while retaining arbitrarily nested dataset packages.
    parents = set()
    for folder, directories, files in os.walk(root):
        directories[:] = [
            d
            for d in directories
            if d
            not in {
                "episodes",
                "native",
                "models",
                "runtimes",
                "source_snapshot",
                "source_snapshots",
                ".runtime",
                "merged",
                "sft",
                "public-crops",
            }
            and not d.startswith("checkpoint-")
        ]
        if "manifest.jsonl" in files:
            for line in (Path(folder) / "manifest.jsonl").read_text().splitlines():
                if line.strip():
                    parents.add(json.loads(line)["split_group"])
    return parents


def revisit_group(group, seed):
    """The same policy-independent prefix is used for all four goal variants."""
    base = copy.deepcopy(group[0]["initial_actions"])
    edits = [a for a in group[0]["oracle_actions"] if a["op"] == "set_parameter"]
    if seed % 2 == 0:
        prefix = copy.deepcopy(edits + edits[:2])
        state_kind = "complete_peer_with_repeats"
    else:
        prefix = copy.deepcopy(edits[: max(1, len(edits) // 2)])
        circles = group[0]["goal_spec"]["circles"]
        index = seed % len(circles)
        prefix.append(
            {
                "op": "set_parameter",
                "target": base[3]["name"],
                "parameter": f"diameter_{index}",
                "value": 2 * circles[index]["radius"] + 1,
            }
        )
        state_kind = "partial_peer_with_wrong_edit"
    current = [2 * c["radius"] for c in group[0]["goal_spec"]["circles"]]
    for a in prefix:
        current[int(a["parameter"].split("_")[-1])] = a["value"]
    tasks = []
    for original in group:
        t = copy.deepcopy(original)
        t["goal_base_actions"] = base
        t["initial_actions"] = base + copy.deepcopy(prefix)
        diameter = next(
            a["value"] for a in t["oracle_actions"] if a["op"] == "set_parameter"
        )
        desired = [
            diameter if i in t["goal_spec"]["selected"] else 2 * c["radius"]
            for i, c in enumerate(t["goal_spec"]["circles"])
        ]
        t["oracle_actions"] = [
            {
                "op": "set_parameter",
                "target": base[3]["name"],
                "parameter": f"diameter_{i}",
                "value": v,
            }
            for i, v in enumerate(desired)
            if not math.isclose(v, current[i], abs_tol=1e-9)
        ] + [{"op": "finish"}]
        t["max_steps"] = 12
        t["public_task"]["max_steps"] = 12
        original_diameter = 2 * t["goal_spec"]["circles"][0]["radius"]
        t["instruction"] += (
            f" The current part may already contain edits. Every UNMARKED hole must have its original diameter {original_diameter:g} mm. Inspect current dimensions: skip values already correct and finish immediately when all requested and preserved dimensions are correct."
        )
        t["public_task"]["instruction"] = t["instruction"]
        t["curriculum_state"] = state_kind
        tasks.append(t)
    return tasks


def finalize_group(group, phase):
    ids = {
        t["task_id"]: "v-"
        + canonical_digest([t["task_id"], "release-data-v1", phase])[:24]
        for t in group
    }
    for t in group:
        old = t["task_id"]
        t["task_id"] = ids[old]
        t["condition_group"] += "-release-" + phase
        t["curriculum_phase"] = phase
        t["counterfactual_goal_images"] = [
            f"episodes/{ids[Path(p).parent.name]}/goal.png"
            for p in t["counterfactual_goal_images"]
        ]
    return group


def specs(train_parents=100, validation_parents=30, denied=()):
    if (
        min(train_parents, validation_parents) < 5
        or train_parents % 5
        or validation_parents % 5
    ):
        raise ValueError("Parent counts must be positive multiples of five")
    remaining = {
        (split, count): n // 5
        for split, n in [("train", train_parents), ("validation", validation_parents)]
        for count in range(2, 7)
    }
    denied = set(denied)
    tasks = []
    for seed in range(7000000, 7100000):
        count = 2 + seed % 5
        diameter = [9.525, 12.7, 15.875][seed % 3]
        public = f"{diameter / 25.4:g} inches" if seed % 2 else f"{diameter:g} mm"
        group = make_visual_group(
            seed,
            selected_count=count,
            target_diameter=diameter,
            public_diameter=public,
            entity_labels=True,
            shuffle_edit_order=False,
        )
        split = group[0]["split"]
        parent = group[0]["split_group"]
        if (
            (split, count) not in remaining
            or not remaining[split, count]
            or parent in denied
        ):
            continue
        denied.add(parent)
        remaining[split, count] -= 1
        for t in group:
            t["goal_spec"]["angle"] = 0
        tasks.extend(finalize_group(copy.deepcopy(group), "clean"))
        tasks.extend(finalize_group(revisit_group(group, seed), "revisit"))
        if not any(remaining.values()):
            break
    if any(remaining.values()):
        raise ValueError("Insufficient fresh natural-split parents")
    return tasks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train-parents", type=int, default=100)
    p.add_argument("--validation-parents", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.workers < 1:
        p.error("Positive workers required")
    tasks = specs(a.train_parents, a.validation_parents, known_parents())
    from .factory import generate_dataset
    from .visual_audit import audit_visual

    summary = generate_dataset(a.output, task_specs=tasks, workers=a.workers)
    root = Path(a.output)
    summary.update(
        benchmark="visual-grounding-release-v1",
        supported_training_exports=["train.feedback.jsonl"],
        test_opened=False,
        parent_counts=dict(
            Counter(t["split"] for t in {t["split_group"]: t for t in tasks}.values())
        ),
        phase_counts=dict(Counter(t["curriculum_phase"] for t in tasks)),
        scope="Fresh natural-split parents; shared-state four-way counterfactuals; IDs and canonical camera. Clean tasks have 8 steps, revisit tasks 12. No TEST task geometry.",
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_visual(root)
    (root / "information-audit.json").write_text(json.dumps(report, indent=2))
    if summary["rejected"] or not report["ok"]:
        raise RuntimeError(str(report["errors"][:5]))
    print(
        json.dumps(
            {
                "accepted": summary["accepted"],
                "parents": summary["parent_counts"],
                "audit_ok": report["ok"],
                "feedback_rows": summary["counts"]["train.feedback"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
