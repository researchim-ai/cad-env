"""TRAIN-only counterfactual repair data seeded by real policy mistakes.

One incorrect, natively executed policy prefix is shared by all four target
variants of a board. This prevents the damaged state itself from revealing
which image target is correct. Teachers restore unmarked original dimensions
and apply selected edits; every trajectory is revalidated in FreeCAD.
"""

import argparse
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .factory import generate_dataset
from .task_identity import load_private_task
from .visual_audit import audit_visual
from .visual_bandit import VisualBanditReward
from .visual_tasks import canonical_digest


def generate_repairs(dataset, calibration, output, *, workers=4, max_parents=32):
    root, source, out = (Path(p).resolve() for p in (dataset, calibration, output))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new repair corpus directory")
    evidence = json.loads((source / "calibration.json").read_text())
    if evidence["status"] != "complete":
        raise ValueError("Complete policy calibration required")
    if (
        evidence["dataset_sha256"]
        != hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Calibration used another dataset")
    if not audit_visual(root)["ok"]:
        raise ValueError("Source visual corpus failed its audit")
    by_group = defaultdict(list)
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["accepted"] and entry["split"] == "train":
            t = load_private_task(
                root / "episodes", entry["task_id"], require_train=True
            )
            by_group[t["condition_group"]].append(t)
    scores = [json.loads(l) for l in (source / "scores.jsonl").read_text().splitlines()]
    checker = VisualBanditReward(
        root, out.parent / (out.name + "-prefix-checks"), workers=workers
    )
    tasks, used_parents = [], set()
    origins = []
    source_sha = hashlib.sha256((source / "scores.jsonl").read_bytes()).hexdigest()
    for row in scores:
        if row["task_success"] or not row["execution_ok"]:
            continue
        original = load_private_task(
            root / "episodes", row["task_id"], require_train=True
        )
        if original["split_group"] in used_parents:
            continue
        checked = checker.score(row["task_id"], row["completion"])
        if checked["task_success"] or not checked["execution_ok"]:
            raise ValueError(
                "Recorded policy error failed independent native revalidation"
            )
        prefix = json.loads(row["completion"])
        base = copy.deepcopy(original["initial_actions"])
        if (
            prefix["op"] != "set_parameter"
            or prefix["target"] != base[3]["name"]
            or not prefix["parameter"].startswith("diameter_")
        ):
            continue
        index = int(prefix["parameter"].removeprefix("diameter_"))
        current = [h["radius"] for h in base[3]["circles"]]
        current[index] = prefix["value"] / 2
        original_radii = {h["radius"] for h in base[3]["circles"]}
        if len(original_radii) != 1:
            raise ValueError("Repair instruction requires a public original diameter")
        original_diameter = 2 * next(iter(original_radii))
        siblings = by_group[original["condition_group"]]
        if len(siblings) != 4:
            raise ValueError("Expected a complete TRAIN counterfactual group")
        prefix_sha = canonical_digest(prefix)
        identifiers = {}
        group_tasks = []
        for sibling in siblings:
            if sibling["initial_actions"] != base:
                raise ValueError("Counterfactual boards differ")
            t = copy.deepcopy(sibling)
            old_id = t["task_id"]
            t["task_id"] = (
                "v-" + canonical_digest([old_id, "on-policy-repair", prefix_sha])[:24]
            )
            identifiers[old_id] = t["task_id"]
            t["family"] = "visual_repair"
            t["mode"] = "edit"
            t["goal_base_actions"] = copy.deepcopy(base)
            t["initial_actions"] = copy.deepcopy(base) + [copy.deepcopy(prefix)]
            desired = [h["radius"] for h in t["reference_actions"][3]["circles"]]
            edits = [
                {
                    "op": "set_parameter",
                    "target": base[3]["name"],
                    "parameter": f"diameter_{i}",
                    "value": 2 * radius,
                }
                for i, radius in enumerate(desired)
                if not math.isclose(radius, current[i], abs_tol=1e-9)
            ]
            random.Random(t["task_id"]).shuffle(edits)
            t["oracle_actions"] = edits + [{"op": "finish"}]
            target = sibling["oracle_actions"][0]["value"]
            instruction = (
                f"Repair the current parametric part. Set every hole selected by a colored crosshair in the goal image to diameter {target:g} mm. "
                f"Every UNMARKED hole must have its original diameter {original_diameter:g} mm. "
                "The current part may contain an earlier wrong edit. Preserve the feature history and all other dimensions. Return one action at a time, then finish."
            )
            t["instruction"] = instruction
            t["public_task"].update(instruction=instruction, mode="edit")
            t["condition_group"] = (
                original["condition_group"] + "-repair-" + prefix_sha[:16]
            )
            t["evaluation_tier"] = "on_policy_repair"
            t["repair_provenance"] = {
                "source_scores_sha256": source_sha,
                "source_task_id": row["task_id"],
                "source_candidate": row["candidate"],
                "prefix_sha256": prefix_sha,
                "target_task_id": old_id,
            }
            t.pop("goal_images", None)
            group_tasks.append(t)
        for t, sibling in zip(group_tasks, siblings):
            old_peer = Path(sibling["counterfactual_goal_images"][0]).parent.name
            t["counterfactual_goal_images"] = [
                f"episodes/{identifiers[old_peer]}/goal.png"
            ]
        tasks.extend(group_tasks)
        used_parents.add(original["split_group"])
        origins.append(
            {
                "source_task_id": row["task_id"],
                "candidate": row["candidate"],
                "prefix_sha256": prefix_sha,
            }
        )
        if len(used_parents) >= max_parents:
            break
    if not tasks:
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "no_native_policy_errors",
            "tasks": 0,
            "note": "No accepted incorrect policy prefix was fabricated to create repair examples",
        }
        (out / "repair-generation.json").write_text(json.dumps(result, indent=2))
        return result
    summary = generate_dataset(
        out,
        task_specs=tasks,
        families=("visual_repair",),
        modes=("edit",),
        workers=workers,
    )
    summary.update(
        benchmark="visual-grounding-repair-v1",
        source_model_identity=evidence["model_identity"],
        source_scores_sha256=source_sha,
        source_policy_prefixes=origins,
        supported_training_exports=["train.feedback.jsonl"],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_visual(out)
    (out / "information-audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["ok"] or summary["rejected"]:
        raise RuntimeError("Counterfactual native repair corpus failed validation")
    result = {
        "status": "complete",
        "tasks": summary["accepted"],
        "parent_boards": len(used_parents),
        "feedback_rows": summary["counts"]["train.feedback"],
        "train_only": True,
        "nonvisual_ceiling": audit["greedy_no_image_success_ceiling"],
        "source_scores_sha256": source_sha,
        "note": "Repair data generation only; no new optimizer update is implied",
    }
    (out / "repair-generation.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-parents", type=int, default=32)
    args = p.parse_args()
    if args.workers < 1 or args.max_parents < 1:
        p.error("Positive workers and max-parents required")
    print(
        json.dumps(
            generate_repairs(
                args.dataset,
                args.calibration,
                args.output,
                workers=args.workers,
                max_parents=args.max_parents,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
