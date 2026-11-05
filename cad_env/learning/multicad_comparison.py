"""Strict paired comparisons for multi-engine full-agent reports."""

import hashlib
import json

from .metrics import paired_comparison


def normalize(report):
    if report.get("status") != "complete":
        raise ValueError("Only complete runs may be compared")
    rows = report["episodes"]
    ids = [r["task_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Expected unique complete episodes")
    return {
        "status": "complete",
        "dataset_digest": report["dataset_sha256"],
        "protocol_digest": hashlib.sha256(
            json.dumps(report["protocol"], sort_keys=True).encode()
        ).hexdigest(),
        "task_ids": sorted(ids),
        "episodes": [{**r, "attempt": 0} for r in rows],
    }


def compare(before, after):
    if before["split"] != after["split"] or before["engine"] != after["engine"]:
        raise ValueError("Mismatched split/engine")
    result = paired_comparison(normalize(before), normalize(after))
    result["before_success"] = before["metrics"]["success_at_1"]
    result["after_success"] = after["metrics"]["success_at_1"]
    return result


def combined(reports):
    pairs = [(normalize(a), normalize(b)) for a, b in reports]
    for a, b in pairs:
        if any(a[k] != b[k] for k in ["dataset_digest", "protocol_digest", "task_ids"]):
            raise ValueError("Incompatible paired run")

    def pack(side):
        items = [p[side] for p in pairs]
        rows = [row for item in items for row in item["episodes"]]
        return {
            "status": "complete",
            "dataset_digest": hashlib.sha256(
                json.dumps([i["dataset_digest"] for i in items]).encode()
            ).hexdigest(),
            "protocol_digest": hashlib.sha256(
                json.dumps([i["protocol_digest"] for i in items]).encode()
            ).hexdigest(),
            "task_ids": sorted(r["task_id"] for r in rows),
            "episodes": rows,
        }

    return paired_comparison(pack(0), pack(1))


def image_effect(full, control):
    if control.get("image_condition") != "no_goal" or full.get(
        "model_identity"
    ) != control.get("model_identity"):
        raise ValueError("Matched model and explicit no-goal condition required")
    result = compare(control, full)
    slices = {}
    for mode in ["create", "edit"]:
        pair = []
        for report in [control, full]:
            rows = [e for e in report["episodes"] if e["mode"] == mode]
            if not rows:
                break
            pair.append(
                {
                    **report,
                    "episodes": rows,
                    "metrics": {
                        "success_at_1": sum(e["task_success"] for e in rows) / len(rows)
                    },
                }
            )
        if len(pair) == 2:
            slices[mode] = compare(*pair)
    return {
        "overall": result,
        "modes": slices,
        "interpretation": "Matched minus no-goal native success; current feedback retained",
    }
