"""Offline diagnostics separating selected-entity errors from numeric/unit errors."""

import argparse
import json
import math
import re
from pathlib import Path

from .protocol import digest


def analyze(dataset, evaluation):
    root, output = Path(dataset).resolve(), Path(evaluation).resolve()
    report = json.loads((output / "evaluation.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if report.get("status") != "complete" or report.get("limited_evaluation"):
        raise ValueError("Complete unlimited evaluation required")
    if report["dataset_digest"] != digest(summary):
        raise ValueError("Dataset identity mismatch")
    rows = [r for r in report["episodes"] if r["attempt"] == 0]
    if len(rows) != len(report["task_ids"]) or {r["task_id"] for r in rows} != set(
        report["task_ids"]
    ):
        raise ValueError("Missing or duplicated first attempts")
    counts = {
        "tasks": len(rows),
        "native_success": 0,
        "first_action_correct_entity": 0,
        "first_action_correct_value": 0,
        "episodes_with_unrequested_edits": 0,
        "finish_before_requested_values": 0,
        "inch_tasks": 0,
        "inch_literal_used_as_mm": 0,
    }
    for row in rows:
        task_path = (root / "episodes" / row["task_id"] / "task.json").resolve()
        trajectory = (output / row["trajectory"]).resolve()
        if not task_path.is_relative_to(
            root / "episodes"
        ) or not trajectory.is_relative_to(output):
            raise ValueError("Unsafe artifact path")
        task = json.loads(task_path.read_text())
        expected = {
            (a["target"], a["parameter"]): a["value"]
            for a in task["oracle_actions"]
            if a["op"] == "set_parameter"
        }
        if not expected:
            raise ValueError("This diagnostic requires visual parameter-edit tasks")
        transitions = [json.loads(l) for l in trajectory.read_text().splitlines()]
        if not transitions:
            raise ValueError("Missing trajectory")
        actions = [t["action"] for t in transitions]
        first = actions[0]

        def close(a, b):
            return (
                isinstance(a, (int, float))
                and not isinstance(a, bool)
                and math.isfinite(a)
                and math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
            )

        is_edit = first.get("op") == "set_parameter"
        counts["first_action_correct_entity"] += (
            is_edit and (first.get("target"), first.get("parameter")) in expected
        )
        counts["first_action_correct_value"] += is_edit and any(
            close(first.get("value"), v) for v in expected.values()
        )
        accepted = [
            t["action"]
            for t in transitions
            if t.get("info", {}).get("execution_ok")
            and t["action"].get("op") == "set_parameter"
        ]
        writes = {
            (a.get("target"), a.get("parameter")): a.get("value") for a in accepted
        }
        counts["episodes_with_unrequested_edits"] += any(
            key not in expected for key in writes
        )
        counts["finish_before_requested_values"] += actions[-1].get(
            "op"
        ) == "finish" and any(not close(writes.get(k), v) for k, v in expected.items())
        unit = re.search(r"to diameter ([0-9.]+) inches", task["instruction"])
        if unit:
            counts["inch_tasks"] += 1
            counts["inch_literal_used_as_mm"] += is_edit and close(
                first.get("value"), float(unit[1])
            )
        counts["native_success"] += bool(row["task_success"])
    return {
        "counts": counts,
        "first_action_entity_rate": counts["first_action_correct_entity"] / len(rows),
        "first_action_value_rate": counts["first_action_correct_value"] / len(rows),
        "scope": "Offline diagnostics only. Unrequested edits are historical writes, not proof of final protected-parameter damage; native task success remains authoritative. Private labels are never policy inputs.",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = analyze(args.dataset, args.evaluation)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
