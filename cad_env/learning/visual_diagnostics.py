"""Offline visual-cohort diagnostics. Private targets never enter policy prompts."""

import json
from collections import defaultdict
from pathlib import Path

from .metrics import paired_comparison
from .protocol import digest


def compare_slices(root, before, after):
    root = Path(root)
    expected = digest(json.loads((root / "summary.json").read_text()))
    # This validates completeness/protocol/lineage before producing any slices.
    overall = paired_comparison(before, after)
    if before["dataset_digest"] != expected:
        raise ValueError("Diagnostics dataset differs from evaluation")
    buckets = defaultdict(set)
    for task_id in before["task_ids"]:
        task = json.loads((root / "episodes" / task_id / "task.json").read_text())
        spec = task["goal_spec"]
        buckets["labels:" + ("yes" if spec.get("entity_labels") else "no")].add(task_id)
        buckets[
            "units:" + ("inches" if " inches" in task["instruction"] else "mm")
        ].add(task_id)
        buckets[f"edits:{len(spec['selected'])}"].add(task_id)
        buckets["tier:" + task.get("evaluation_tier", "iid")].add(task_id)
    result = {
        "overall": overall,
        "slices": {},
        "scope": "Exploratory cohort diagnostics; shared physical parents and multiple slice comparisons, not independent benchmarks",
    }
    for name, ids in sorted(buckets.items()):
        reports = []
        rates = []
        for report in [before, after]:
            rows = [r for r in report["episodes"] if r["task_id"] in ids]
            reports.append({**report, "task_ids": sorted(ids), "episodes": rows})
            rates.append(sum(r["task_success"] for r in rows) / len(rows))
        result["slices"][name] = {
            **paired_comparison(*reports),
            "before_success": rates[0],
            "after_success": rates[1],
        }
    return result
