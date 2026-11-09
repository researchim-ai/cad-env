"""Read-only feature diagnostics on a snapshot of completed paired episodes.

Cut signatures ignore names/order but are only a program-level proxy: supports,
overlap, and native geometric equivalence remain the native evaluator's job.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def cuts(actions):
    sketches, result = {}, []
    for a in actions:
        if a["op"] == "sketch_circles":
            sketches[a["name"]] = a
        elif a["op"] == "pocket" and a.get("profile") in sketches:
            sketch = sketches[a["profile"]]
            for c in sketch["circles"]:
                result.append(
                    {
                        "name": a["name"],
                        "counterbore": "CB" in a["name"],
                        "signature": tuple(
                            round(float(v), 6)
                            for v in (c["x"], c["y"], c["radius"], a["length"])
                        ),
                    }
                )
    return result


def compare(teacher, student):
    expected, actual = cuts(teacher), cuts(student)
    available = Counter(x["signature"] for x in actual)
    matched, missing = [], []
    for x in expected:
        if available[x["signature"]]:
            matched.append(x)
            available[x["signature"]] -= 1
        else:
            missing.append(x)
    return {
        "expected_cuts": len(expected),
        "matched_cuts": len(matched),
        "expected_counterbores": sum(x["counterbore"] for x in expected),
        "matched_counterbores": sum(x["counterbore"] for x in matched),
        "expected_primary_cuts": sum(not x["counterbore"] for x in expected),
        "matched_primary_cuts": sum(not x["counterbore"] for x in matched),
        "missing": missing,
        "unmatched_student_cuts": sum(available.values()),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root, baseline = Path(args.run), Path(args.baseline)
    status = json.loads((root / "evaluation/status.json").read_text())
    pairs = {
        x["task_id"]: x
        for x in json.loads((root / "evaluation/selection.json").read_text())
    }
    before = json.loads((baseline / "status.json").read_text())["models"][
        "release-candidate"
    ]["episodes"]
    before = {(x["task_id"], x["standard"]): x for x in before}
    cases, errors = [], Counter()
    for model in status["models"].values():
        for row in model["episodes"]:
            if row["status"] != "complete":
                continue
            pair = pairs[row["task_id"]]
            task = json.loads(
                Path(pair["representations"][row["standard"]]["task"]).read_text()
            )
            detail = compare(task["reference_actions"], row["committed_actions"])
            case = root / "evaluation/tuned-iso-gost" / row["standard"] / row["task_id"]
            for trace in (case / "native").rglob("trajectory.jsonl"):
                for line in trace.read_text().splitlines():
                    item = json.loads(line)
                    # Trace schema keeps execution outcome inside info.
                    info = item.get("info", {})
                    error = info.get("error") or item.get("observation_after", {}).get(
                        "error"
                    )
                    if error:
                        errors[str(error)] += 1
            old = before[row["task_id"], row["standard"]]
            cases.append(
                {
                    "task_id": row["task_id"],
                    "standard": row["standard"],
                    "stop": row["stop_reason"],
                    "success": row["task_success"],
                    "steps": row["steps"],
                    "invalid": row["invalid_actions"],
                    "geometry_before": old["validation"]["geometry_score"],
                    "geometry_after": row["validation"]["geometry_score"],
                    **detail,
                }
            )
    selected = json.loads((root / "sampling-plan.json").read_text())["selected"]
    summary = {
        "completed_snapshot": len(cases),
        "evaluation_status": status["status"],
        "successes": sum(x["success"] for x in cases),
        "mean_geometry_before_same_cases": sum(x["geometry_before"] for x in cases)
        / len(cases)
        if cases
        else None,
        "mean_geometry_after_same_cases": sum(x["geometry_after"] for x in cases)
        / len(cases)
        if cases
        else None,
        "training_rows": len(selected),
        "training_recovery_rows": sum(x["recovery"] for x in selected),
        "training_cb_sketch_rows": sum(
            x["counterbore"] and x["operation"] == "sketch_circles" for x in selected
        ),
        "training_cb_pocket_rows": sum(
            x["counterbore"] and x["operation"] == "pocket" for x in selected
        ),
        "trace_error_messages": dict(errors.most_common(15)),
        "cuts": {
            key: sum(x[key] for x in cases)
            for key in (
                "expected_cuts",
                "matched_cuts",
                "expected_counterbores",
                "matched_counterbores",
                "expected_primary_cuts",
                "matched_primary_cuts",
            )
        },
        "scope": "Completed-case snapshot, not final benchmark. Program cut signatures ignore names/order, not a replacement for native evaluation.",
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
