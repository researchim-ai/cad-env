"""Summarize a complete paired CAD rollout pilot, keeping perception probes separate."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def analyze(root):
    root = Path(root).resolve()
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Only a complete, infrastructure-error-free run is reportable")
    selection = json.loads((root / "selection.json").read_text())
    targets = {
        p["task_id"]: json.loads(Path(p["representations"]["iso"]["task"]).read_text())
        for p in selection
    }
    report = {
        "status": "complete",
        "independent_geometries": len(selection),
        "protocol": status["protocol"],
        "models": {},
        "scope": "Small bounded development pilot on synthetic nominal-geometry inputs. Numerical component diagnostics are exploratory and do not establish visual grounding without image ablations.",
    }
    for label, model in status["models"].items():
        rows = model["episodes"]
        if len(rows) != len(selection) * 2 or any(
            r["status"] != "complete" for r in rows
        ):
            raise ValueError("Incomplete model episodes")
        values = {}
        for standard in ("iso", "gost"):
            episodes = [r for r in rows if r["standard"] == standard]
            errors = Counter()
            parseable = proposed = 0
            rectangle = {"applicable": 0, "proposed": 0, "exact_first_proposal": 0}
            circle = {"applicable": 0, "proposed": 0, "exact_first_radius": 0}
            examples = {}
            for row in episodes:
                target = targets[row["task_id"]]
                folder = root / label / standard / row["task_id"]
                actions = []
                for line in (folder / "responses.jsonl").read_text().splitlines():
                    proposed += 1
                    response = json.loads(line)["response"]
                    try:
                        action = json.loads(response)
                    except ValueError:
                        continue
                    parseable += 1
                    if isinstance(action, dict):
                        actions.append(action)
                if "base" in target["parameters"]:
                    rectangle["applicable"] += 1
                    first = next(
                        (a for a in actions if a.get("op") == "sketch_rectangle"), None
                    )
                    if first:
                        rectangle["proposed"] += 1
                        expected = target["parameters"]["base"]
                        rectangle["exact_first_proposal"] += int(
                            first.get("length") == expected["length"]
                            and first.get("width") == expected["width"]
                        )
                else:
                    circle["applicable"] += 1
                    first = next(
                        (a for a in actions if a.get("op") == "sketch_circles"), None
                    )
                    if (
                        first
                        and isinstance(first.get("circles"), list)
                        and first["circles"]
                    ):
                        circle["proposed"] += 1
                        circle["exact_first_radius"] += int(
                            first["circles"][0].get("radius")
                            == target["parameters"]["pads"][0]["radius"]
                        )
                trajectory = Path(row["episode_dir"]) / "trajectory.jsonl"
                for line in trajectory.read_text().splitlines():
                    event = json.loads(line)
                    if event["info"]["execution_ok"]:
                        continue
                    message = event["observation_after"].get("error", "")
                    category = message.split(":", 1)[0]
                    errors[category] += 1
                    examples.setdefault(
                        category,
                        {
                            "task_id": row["task_id"],
                            "action": event["action"],
                            "error": message,
                        },
                    )
            values[standard] = {
                "tasks": len(episodes),
                "successes": sum(r["task_success"] for r in episodes),
                "success_rate": statistics.mean(r["task_success"] for r in episodes),
                "mean_strict_reward": statistics.mean(r["reward"] for r in episodes),
                "valid_final_solids": sum(
                    bool(r["validation"].get("shape_valid")) for r in episodes
                ),
                "mean_native_solid_iou": statistics.mean(
                    r["validation"].get("geometry_score", 0) for r in episodes
                ),
                "stop_reasons": dict(Counter(r["stop_reason"] for r in episodes)),
                "steps": sum(r["steps"] for r in episodes),
                "invalid_actions": sum(r["invalid_actions"] for r in episodes),
                "json_parseable_responses": parseable,
                "total_responses": proposed,
                "median_episode_seconds": statistics.median(
                    r["elapsed_seconds"] for r in episodes
                ),
                "native_errors": dict(errors),
                "error_examples": examples,
                "first_rectangle_dimension_proposal": rectangle,
                "first_flange_circle_radius_proposal": circle,
            }
        indexed = {(r["task_id"], r["standard"]): r for r in rows}
        agreements = Counter()
        for task in targets:
            iso = indexed[task, "iso"]["task_success"]
            gost = indexed[task, "gost"]["task_success"]
            agreements[
                "both_success"
                if iso and gost
                else "iso_only"
                if iso
                else "gost_only"
                if gost
                else "both_fail"
            ] += 1
        report["models"][label] = {
            "profiles": values,
            "paired_outcomes": dict(agreements),
            "checkpoint": model["path"],
            "memory": model.get("memory", {}),
        }
    controls = json.loads((root / "oracle-controls.json").read_text())
    if len(controls) != 6 or not all(r["success"] for r in controls):
        raise ValueError("Native oracle positive controls failed")
    report["native_oracle_controls"] = {"cases": 6, "successes": 6}
    (root / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(
        json.dumps(
            {
                label: {
                    s: {
                        k: v[k]
                        for k in (
                            "tasks",
                            "successes",
                            "mean_native_solid_iou",
                            "invalid_actions",
                        )
                    }
                    for s, v in m["profiles"].items()
                }
                for label, m in result["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
