"""Freeze balanced v2 task plans by natural split, optionally generate paired data."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from cad_env.freecad_v2.__main__ import generate
from cad_env.freecad_v2.compositions import FAMILIES, LEVELS, make_task


def plan(split, count, seed, excluded=()):
    if split not in ("train", "validation", "test") or count < 1:
        raise ValueError("Positive task quota and natural split required")
    selected = []
    seen = set(excluded)
    for family in FAMILIES:
        for level in LEVELS:
            found = 0
            candidate = seed
            while found < count:
                task = make_task(family, level, candidate)
                if task["split"] == split and task["split_group"] not in seen:
                    selected.append({"seed": candidate, "task": task})
                    seen.add(task["split_group"])
                    found += 1
                candidate += 1
                if candidate - seed > max(10000, count * 1000):
                    raise ValueError("Prespecified seed search exhausted")
    return selected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], required=True)
    p.add_argument("--per-family-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--exclude-plan", type=Path, action="append", default=[])
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    excluded = {
        r["task"]["split_group"]
        for path in args.exclude_plan
        for r in json.loads(path.read_text())["selected"]
    }
    selected = plan(args.split, args.per_family_level, args.seed, excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "planned",
        "split": args.split,
        "selected": selected,
        "profiles": ["iso", "gost"],
        "selection_rule": "First unique geometry lineages in sequential seed search with the requested natural split; no model/native outcome selection.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = args.output / "task-plan.json"
    path.write_text(json.dumps(state, indent=2))
    shutil.copyfile(__file__, args.output / "plan-runner.py")
    if not args.plan_only:
        by_key = {}
        for row in selected:
            task = row["task"]
            by_key.setdefault(task["family"] + ":" + task["difficulty"], []).append(
                task
            )

        def factory(key, index):
            return by_key[key][index]

        try:
            summary = generate(
                args.output / "dataset",
                args.per_family_level,
                0,
                list(by_key),
                task_factory=factory,
                drawings=True,
            )
            state.update(
                status="complete",
                accepted=summary["accepted"],
                attempted=summary["attempted"],
            )
        except Exception as exc:
            state.update(status="failed", error=repr(exc))
            raise
        finally:
            path.write_text(json.dumps(state, indent=2))
    print(
        json.dumps(
            {
                "status": state["status"],
                "geometries": len(selected),
                "drawing_bundles": 2 * len(selected),
                "split": args.split,
            }
        )
    )


if __name__ == "__main__":
    main()
