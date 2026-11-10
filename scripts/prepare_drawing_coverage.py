"""Prepare paired coverage indices; no model loading or training is performed."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from cad_env.drawings.export import sha
from cad_env.learning.drawing_coverage import select_decisions


def prepare(paired, heldout, count, seed):
    pairs = [json.loads(s) for s in (paired / "pairs.jsonl").read_text().splitlines()]
    excluded = {p["split_group"] for p in pairs if p["split"] != "train"}
    for path in heldout:
        excluded.update(x["split_group"] for x in json.loads(path.read_text()))
    by_task = {p["task_id"]: p for p in pairs}
    cache, sources, decisions = {}, {}, []
    for pair in pairs:
        if pair["split"] != "train":
            continue
        if pair["split_group"] in excluded:
            raise ValueError("TRAIN/held-out lineage overlap")
        paired_rows = []
        for standard in ("iso", "gost"):
            root = Path(pair["representations"][standard]["dataset"])
            if root not in cache:
                export = root / "exports/train.feedback.jsonl"
                grouped = defaultdict(list)
                for i, line in enumerate(export.read_text().splitlines()):
                    row = json.loads(line)
                    parent = by_task[row["task_id"]]
                    if parent["split"] != "train" or row["split_group"] in excluded:
                        raise ValueError("Held-out row in TRAIN export")
                    if row["split_group"] != parent["split_group"]:
                        raise ValueError("Row lineage mismatch")
                    grouped[row["task_id"]].append((i, row))
                cache[root] = grouped
                sources[str(root)] = {
                    "standard": standard,
                    "export_sha256": sha(export),
                }
            task = json.loads(
                Path(pair["representations"][standard]["task"]).read_text()
            )
            if task["split"] != "train" or task["split_group"] != pair["split_group"]:
                raise ValueError("Task lineage mismatch")
            paired_rows.append((root, cache[root][pair["task_id"]], task))
        left, right = paired_rows
        if not left[1] or len(left[1]) != len(right[1]):
            raise ValueError("Missing paired decisions")
        for (li, a), (ri, b) in zip(left[1], right[1]):
            for key in ("step_id", "completion", "recovery", "split_group"):
                if a.get(key) != b.get(key):
                    raise ValueError("ISO/GOST causal decisions differ")
            action = json.loads(a["completion"][0]["content"][0]["text"])
            decisions.append(
                {
                    "task_id": pair["task_id"],
                    "split_group": pair["split_group"],
                    "split": "train",
                    "family": pair["family"],
                    "step_id": a["step_id"],
                    "operation": action["op"],
                    "recovery": a.get("recovery", False),
                    "counterbore": "CB" in action.get("name", "")
                    or "CB" in action.get("profile", ""),
                    "progress": a["step_id"]
                    / max(1, len(left[2]["reference_actions"]) - 1),
                    "representations": {
                        "iso": {"source": str(left[0]), "row_index": li},
                        "gost": {"source": str(right[0]), "row_index": ri},
                    },
                }
            )
    selected, report = select_decisions(decisions, count, seed)
    return {
        "schema": "cad-env.paired-coverage-plan.v1",
        "seed": seed,
        "training_rows": 2 * count,
        "standard_rows": {"iso": count, "gost": count},
        "sources": sources,
        "selected": selected,
        "report": report,
        "heldout_sha256": {str(p): sha(p) for p in heldout},
        "pairs_sha256": sha(paired / "pairs.jsonl"),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--paired", type=Path, required=True)
    p.add_argument("--heldout", type=Path, nargs="+", required=True)
    p.add_argument("--decisions", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = prepare(args.paired, args.heldout, args.decisions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(
        json.dumps({"training_rows": plan["training_rows"], **plan["report"]}, indent=2)
    )


if __name__ == "__main__":
    main()
