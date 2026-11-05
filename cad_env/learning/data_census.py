"""Count independent TRAIN geometries, feedback repetitions and feature graphs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def graph_signature(actions):
    names = {}
    graph = []
    for action in actions:
        op = action["op"]
        if op == "finish":
            continue
        node = {"op": op}
        for field in ["body", "profile", "support", "target"]:
            if field in action:
                node[field] = names.get(action[field], "external")
        if "circles" in action:
            node["circle_count"] = len(action["circles"])
        if "parameter" in action:
            node["parameter"] = action["parameter"]
        if "name" in action:
            names[action["name"]] = len(graph)
        graph.append(node)
    return hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()


def census(root):
    root = Path(root)
    entries = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    accepted = [r for r in entries if r["accepted"] and r["split"] == "train"]
    rows = [
        json.loads(line)
        for line in (root / "exports/train.feedback.jsonl").read_text().splitlines()
    ]
    graphs = Counter()
    for entry in accepted:
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        graphs[graph_signature(task["reference_actions"])] += 1
    return {
        "root": str(root.resolve()),
        "train_episodes": len(accepted),
        "train_physical_groups": len({r["split_group"] for r in accepted}),
        "train_feedback_rows": len(rows),
        "feature_graph_signatures": len(graphs),
        "graph_episode_counts": dict(graphs),
        "note": "Signature ignores dimensions and identifier names; keeps operation order, feature dependencies and circle count. It is a structural diagnostic, not BRep equivalence or a semantic diversity metric.",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = [census(root) for root in args.dataset]
    Path(args.output).write_text(json.dumps(result, indent=2))
    for row in result:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["graph_episode_counts", "note"]
                }
            )
        )


if __name__ == "__main__":
    main()
