"""One entry point for nine procedural CAD families with paired ISO/GOST exports."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.batch import write_json
from cad_env.drawings.export import sha


def commands(args):
    root = Path(args.output).resolve()
    shared = ["--split", args.split, "--levels", *args.levels, "--jobs", str(args.jobs)]
    for source in args.exclude_pairs:
        shared.extend(["--exclude-pairs", source])
    if args.plan_only:
        shared.append("--plan-only")
    result = []
    if args.suite in ("all", "curriculum"):
        result.append(
            (
                "curriculum",
                [
                    sys.executable,
                    "-u",
                    "scripts/generate_curriculum_drawings.py",
                    "--output",
                    str(root / "curriculum"),
                    "--per-level",
                    str(args.per_family_level * 3),
                    "--seed",
                    str(args.seed),
                    *shared,
                ],
            )
        )
    if args.suite in ("all", "industrial"):
        result.append(
            (
                "industrial",
                [
                    sys.executable,
                    "-u",
                    "scripts/generate_industrial_drawings.py",
                    "--output",
                    str(root / "industrial"),
                    "--per-family-level",
                    str(args.per_family_level),
                    "--seed",
                    str(args.seed + 10000000),
                    *shared,
                ],
            )
        )
    return result


def merge_indices(root, suites):
    pairs = []
    for name in suites:
        folder = root / name / "paired"
        if json.loads((folder / "summary.json").read_text())["status"] != "complete":
            raise ValueError("A child corpus is incomplete")
        pairs.extend(
            json.loads(line)
            for line in (folder / "pairs.jsonl").read_text().splitlines()
        )
    if len({p["split_group"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate physical lineage across suites")
    if len({p["task_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate task identity across suites")
    datasets = sorted(
        {p["representations"][std]["dataset"] for p in pairs for std in ("iso", "gost")}
    )
    target = root / "paired"
    target.mkdir(exist_ok=True)
    temp = target / "pairs.tmp"
    temp.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    temp.replace(target / "pairs.jsonl")
    write_json(
        target / "summary.json",
        {
            "status": "complete",
            "independent_geometries": len(pairs),
            "documents": len(pairs) * 2,
            "family_counts": dict(Counter(p["family"] for p in pairs)),
            "difficulty_counts": dict(Counter(p["difficulty"] for p in pairs)),
            "can_claim_full_conformance": False,
        },
    )
    write_json(
        root / "dataset-sources.json",
        {
            "datasets": datasets,
            "scope": "Pass these profile dataset directories to the existing training loader; top-level root is an orchestration index.",
        },
    )
    return pairs


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--suite", choices=("all", "curriculum", "industrial"), default="all"
    )
    p.add_argument("--per-family-level", type=int, default=1)
    p.add_argument("--split", choices=("train", "validation"), default="train")
    p.add_argument(
        "--levels",
        nargs="+",
        choices=("easy", "medium", "hard"),
        default=["easy", "medium", "hard"],
    )
    p.add_argument("--seed", type=int, default=41000000)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--exclude-pairs", action="append", default=[])
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if min(args.per_family_level, args.jobs) < 1:
        p.error("Positive counts required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "arguments": {k: v for k, v in vars(args).items() if k != "plan_only"},
        "launcher_sha256": sha(__file__),
    }
    request = root / "request.json"
    if request.exists() and json.loads(request.read_text()) != state:
        raise ValueError("Request changed; choose a fresh output directory")
    write_json(request, state)
    jobs = commands(args)
    for name, command in jobs:
        print("Generating suite:", name, flush=True)
        subprocess.run(command, check=True)
    if not args.plan_only:
        rows = merge_indices(root, [name for name, _ in jobs])
        print(
            f"Complete: {len(rows)} geometries / {len(rows) * 2} ISO/GOST documents",
            flush=True,
        )


if __name__ == "__main__":
    main()
