"""Regrade wrong-goal trajectories against the substituted image's native target.

This is a post-hoc diagnostic only. It never changes original-target rewards,
policy observations, training examples, or checkpoint selection criteria.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend

from .rewards import score_completion


def regrade_peer(dataset, evaluation, output, workers=4):
    root, path, out = (Path(p).resolve() for p in (dataset, evaluation, output))
    report = json.loads(path.read_text())
    if (
        report["status"] != "complete"
        or report["protocol"].get("image_condition") != "wrong_goal"
    ):
        raise ValueError("A complete wrong-goal evaluation is required")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new peer diagnostic output")
    out.mkdir(parents=True, exist_ok=True)
    tasks = {}
    by_image = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["accepted"]:
            task = json.loads(
                (root / "episodes" / entry["task_id"] / "task.json").read_text()
            )
            tasks[entry["task_id"]] = task
            key = tuple(task.get("goal_images", []))
            if key:
                if key in by_image:
                    raise ValueError("Ambiguous peer image mapping")
                by_image[key] = task

    def grade(episode):
        original = tasks[episode["task_id"]]
        peer = by_image[tuple(original["counterfactual_goal_images"])]
        if (
            original["condition_group"] != peer["condition_group"]
            or original["task_id"] == peer["task_id"]
        ):
            raise ValueError(
                "Peer must be a different target within the same public-input group"
            )
        trajectory = [
            json.loads(l)
            for l in (path.parent / episode["trajectory"]).read_text().splitlines()
        ]
        actions = [r["action"] for r in trajectory if r["info"].get("execution_ok")]
        result = score_completion(
            peer, json.dumps(actions), out / episode["task_id"], FreeCADBackend()
        )
        return {
            "task_id": episode["task_id"],
            "peer_id": peer["task_id"],
            "original_success": episode["task_success"],
            "peer_success": result["task_success"],
            "peer_reward": result["reward"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(grade, report["episodes"]))
    result = {
        "status": "complete",
        "tasks": len(rows),
        "original_success": sum(r["original_success"] for r in rows) / len(rows),
        "peer_success": sum(r["peer_success"] for r in rows) / len(rows),
        "episodes": rows,
        "note": "Native regrading against substituted image; original evaluation remains unchanged",
    }
    (out / "peer-control.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    print(
        json.dumps(
            regrade_peer(args.dataset, args.evaluation, args.output, args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
