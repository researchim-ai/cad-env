"""Independent offline audit of the saved single-hole native GRPO rewards."""

import argparse
import json
from pathlib import Path

from .task_identity import load_private_task


def audit(run, dataset):
    run, dataset = Path(run), Path(dataset)
    report = json.loads((run / "cad-training-run.json").read_text())
    samples = []
    for path in sorted((run / "native-rollouts").glob("*/*/score.json")):
        score = json.loads(path.read_text())
        task = load_private_task(
            dataset / "episodes", score["task_id"], require_train=True
        )
        selected = task["goal_spec"]["selected"]
        if len(selected) != 1:
            raise ValueError("Audit only supports the declared single-hole contract")
        before = {
            a["name"]: a for a in task["initial_actions"] if a["op"] == "sketch_circles"
        }
        reference = {
            a["name"]: a
            for a in task["reference_actions"]
            if a["op"] == "sketch_circles"
        }
        changes = []
        for name, sketch in before.items():
            for i, circle in enumerate(sketch["circles"]):
                target = reference[name]["circles"][i]
                if circle["radius"] != target["radius"]:
                    changes.append(
                        {
                            "op": "set_parameter",
                            "target": name,
                            "parameter": f"diameter_{i}",
                            "value": 2 * target["radius"],
                        }
                    )
        if len(changes) != 1:
            raise ValueError("Unexpected reference changes")
        try:
            completion = score["completion"]
            if isinstance(completion, list):
                completion = completion[0]["content"]
            action = json.loads(completion)
        except (ValueError, TypeError, KeyError):
            action = None
        # Offline check reads private geometry solely to audit saved rewards;
        # these values are never added to policy observations or training prompts.
        expected = action == changes[0]
        samples.append(
            {
                "task_id": task["task_id"],
                "split_group": task["split_group"],
                "native_reward": score["reward"],
                "independent_success": expected,
                "agrees": (score["reward"] == float(expected)),
            }
        )
    if not samples:
        raise ValueError("No saved native rollouts")
    return {
        "scope": "offline reward audit, not a learning benchmark",
        "samples": len(samples),
        "unique_tasks": len({s["task_id"] for s in samples}),
        "unique_physical_groups": len({s["split_group"] for s in samples}),
        "successes": sum(s["native_reward"] for s in samples),
        "disagreements": [s for s in samples if not s["agrees"]],
        "gradient_signal": report.get("gradient_signal"),
        "arguments": report["arguments"],
        "learning_claim_supported": False,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ["run", "dataset", "output"]:
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    result = audit(args.run, args.dataset)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["disagreements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
