"""Paired image-intervention statistics under an otherwise identical contract."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def paired_image_control(full, control, *, task_ids=None, seed=42, resamples=2000):
    if resamples < 100:
        raise ValueError("At least 100 cluster bootstrap samples required")
    for report in (full, control):
        if report.get("status") != "complete" or report.get("limited_evaluation"):
            raise ValueError("Controls require complete, unlimited evaluations")
    for field in ("dataset_digest", "task_ids", "model_identity"):
        if full[field] != control[field]:
            raise ValueError("Image control differs in " + field)
    if full["protocol"].get("image_condition", "full") != "full":
        raise ValueError("First report must use matched images")
    condition = control["protocol"].get("image_condition")
    if condition not in ("no_goal", "blank_goal", "wrong_goal"):
        raise ValueError("An explicit image intervention is required")
    a, b = dict(full["protocol"]), dict(control["protocol"])
    a.pop("image_condition", None)
    b.pop("image_condition", None)
    if a != b:
        raise ValueError(
            "Decoding, runtime, prompt, split or budgets changed alongside images"
        )
    selected = set(task_ids) if task_ids is not None else set(full["task_ids"])
    if not selected or not selected.issubset(full["task_ids"]):
        raise ValueError("Slice must be a nonempty subset of the frozen cases")
    if a["attempts"] != 1:
        raise ValueError("This control statistic requires one greedy attempt per task")
    rows = []
    for report in (full, control):
        mapping = {
            r["task_id"]: r for r in report["episodes"] if r["task_id"] in selected
        }
        if set(mapping) != selected or len(mapping) != len(
            [r for r in report["episodes"] if r["task_id"] in selected]
        ):
            raise ValueError("Incomplete or duplicated control episodes")
        rows.append(mapping)
    clusters = defaultdict(list)
    for task_id in sorted(selected):
        x, y = rows[0][task_id], rows[1][task_id]
        if x["split_group"] != y["split_group"]:
            raise ValueError("Image control changed parent geometry")
        clusters[x["split_group"]].append(
            float(x["task_success"]) - float(y["task_success"])
        )
    groups = list(clusters.values())
    rng = random.Random(seed)
    sampled = sorted(
        statistics.mean(v for _ in groups for v in rng.choice(groups))
        for _ in range(resamples)
    )
    gap = statistics.mean(v for group in groups for v in group)
    ci = [
        sampled[int(0.025 * resamples)],
        sampled[min(resamples - 1, int(0.975 * resamples))],
    ]
    return {
        "condition": condition,
        "tasks": len(selected),
        "parent_groups": len(groups),
        "matched_success": statistics.mean(
            float(r["task_success"]) for r in rows[0].values()
        ),
        "control_success": statistics.mean(
            float(r["task_success"]) for r in rows[1].values()
        ),
        "matched_minus_control": gap,
        "cluster_bootstrap_95_ci": ci,
        "image_effect_supported": ci[0] > 0 and len(groups) >= 20,
        "seed": seed,
        "resamples": resamples,
        "note": "Image intervention under one model and frozen native contract; parent-geometry bootstrap",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--full", required=True)
    p.add_argument("--control", required=True)
    p.add_argument(
        "--dataset", help="Also report prespecified text/image modality slices"
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    full = json.loads(Path(args.full).read_text())
    control = json.loads(Path(args.control).read_text())
    report = {"overall": paired_image_control(full, control)}
    if args.dataset:
        by_modality = defaultdict(list)
        root = Path(args.dataset).resolve()
        for task_id in full["task_ids"]:
            task = json.loads((root / "episodes" / task_id / "task.json").read_text())
            by_modality[task.get("input_modality", "visual")].append(task_id)
        report["modalities"] = {
            k: paired_image_control(full, control, task_ids=v)
            for k, v in by_modality.items()
        }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
