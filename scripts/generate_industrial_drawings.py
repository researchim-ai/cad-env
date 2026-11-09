"""Scalable native-verified industrial analogues; preserves the running v1 benchmark."""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.batch import quarantine, verify_tasks, write_json
from cad_env.drawings.dataset import audit_technical_dataset, convert_dataset
from cad_env.drawings.export import sha
from cad_env.learning.audit import audit_dataset
from cad_env.learning.factory import generate_dataset
from cad_env.learning.industrial_parts import FAMILIES, LEVELS, make_industrial_task
from scripts.generate_paired_drawings import complete_shard


def select_plan(
    per_family_level=1,
    split="train",
    seed=31000000,
    families=FAMILIES,
    levels=LEVELS,
    shard_size=5,
    excluded=(),
):
    if per_family_level < 1 or shard_size < 1 or split not in ("train", "validation"):
        raise ValueError("Positive counts and train/validation split required")
    if (
        not families
        or set(families) - set(FAMILIES)
        or len(set(families)) != len(families)
    ):
        raise ValueError("Choose distinct supported families")
    if not levels or set(levels) - set(LEVELS) or len(set(levels)) != len(levels):
        raise ValueError("Choose distinct levels")
    seen, shards = set(excluded), []
    for family in families:
        for level in levels:
            tasks = []
            for s in range(seed, seed + max(10000, per_family_level * 1000)):
                t = make_industrial_task(family, s, level)
                if t["split"] == split and t["split_group"] not in seen:
                    tasks.append(t)
                    seen.add(t["split_group"])
                if len(tasks) == per_family_level:
                    break
            if len(tasks) != per_family_level:
                raise ValueError("Sampling window exhausted; use another seed")
            for i in range(0, len(tasks), shard_size):
                shards.append(
                    {
                        "name": f"{family}-{level}-{i // shard_size:05d}",
                        "standard": "iso",
                        "tasks": tasks[i : i + shard_size],
                    }
                )
    return json.loads(
        json.dumps(
            {
                "version": "industrial-analogues-v1",
                "count": per_family_level * len(families) * len(levels),
                "per_family_level": per_family_level,
                "split": split,
                "seed": seed,
                "families": list(families),
                "levels": list(levels),
                "excluded_groups": sorted(set(excluded)),
                "shards": shards,
            }
        )
    )


def primary_shard(root, shard):
    source = root / "shards" / shard["name"] / "source"
    dataset = source.parent / "dataset"
    if source.exists():
        try:
            verify_tasks(source, shard["tasks"])
            if not audit_dataset(source)["ok"]:
                raise ValueError("Existing source failed audit")
        except (ValueError, KeyError, OSError):
            quarantine(root, dataset)
            quarantine(root, source)
    if not source.exists():
        generate_dataset(
            source,
            task_specs=shard["tasks"],
            families=tuple({t["family"] for t in shard["tasks"]}),
            modes=("create",),
            workers=1,
            timeout=180,
            preferences=False,
        )
    verify_tasks(source, shard["tasks"])
    if not audit_dataset(source)["ok"]:
        raise ValueError("Native teacher source failed audit")
    if dataset.exists():
        try:
            info = json.loads((dataset / "summary.json").read_text())
            if (
                info["drawing_standard"] != "iso"
                or not audit_technical_dataset(dataset)["ok"]
            ):
                raise ValueError("Existing drawings failed audit")
        except (ValueError, KeyError, OSError):
            quarantine(root, dataset)
    if not dataset.exists():
        convert_dataset(
            source,
            dataset,
            standard="iso",
            metadata={
                "synthetic_example": True,
                "owner": "CAD-ENV",
                "status": "DRAFT",
                "issue_date": "2026-11-09",
                "document_type": "Part drawing",
            },
        )
    info = json.loads((dataset / "summary.json").read_text())
    if (
        info["accepted"] != len(shard["tasks"])
        or not audit_technical_dataset(dataset)["ok"]
    ):
        raise ValueError("Drawing dataset failed audit")
    result = {
        "name": shard["name"],
        "status": "complete",
        "standard": "iso",
        "tasks": len(shard["tasks"]),
        "dataset": str(dataset.relative_to(root)),
        "summary_sha256": sha(dataset / "summary.json"),
    }
    write_json(source.parent / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--per-family-level", type=int, default=1)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    p.add_argument("--split", choices=("train", "validation"), default="train")
    p.add_argument("--seed", type=int, default=31000000)
    p.add_argument("--shard-size", type=int, default=5)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--exclude-pairs", action="append", default=[])
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("Positive jobs required")
    excluded = set()
    for filename in args.exclude_pairs:
        excluded.update(
            json.loads(x)["split_group"]
            for x in Path(filename).read_text().splitlines()
        )
    plan = select_plan(
        args.per_family_level,
        args.split,
        args.seed,
        tuple(args.families),
        tuple(args.levels),
        args.shard_size,
        excluded,
    )
    root = Path(args.output).resolve()
    primary, paired = root / "primary", root / "paired"
    primary.mkdir(parents=True, exist_ok=True)
    paired.mkdir(exist_ok=True)
    files = [
        Path(__file__),
        Path("cad_env/learning/industrial_parts.py"),
        *Path("cad_env/learning").glob("*.py"),
        Path("scripts/generate_paired_drawings.py"),
        *Path("cad_env/drawings").glob("*.py"),
        *Path("cad_env/agent").glob("*.py"),
    ]
    frozen = {"plan": plan, "sources": {str(f.resolve()): sha(f) for f in files}}
    request = root / "request.json"
    if request.exists() and json.loads(request.read_text()) != frozen:
        raise ValueError("Frozen plan/code differs; use a new output")
    write_json(request, frozen)
    write_json(primary / "plan.json", plan)
    write_json(
        paired / "request.json",
        {"primary": str(primary), "primary_plan_sha256": sha(primary / "plan.json")},
    )
    tasks = [t for s in plan["shards"] for t in s["tasks"]]
    stats = {
        "geometries": len(tasks),
        "documents": len(tasks) * 2,
        "family_counts": dict(Counter(t["family"] for t in tasks)),
        "levels": {
            l: {
                "count": sum(t["difficulty"] == l for t in tasks),
                "action_range": [
                    min(
                        len(t["oracle_actions"]) for t in tasks if t["difficulty"] == l
                    ),
                    max(
                        len(t["oracle_actions"]) for t in tasks if t["difficulty"] == l
                    ),
                ],
            }
            for l in args.levels
        },
        "split": args.split,
        "can_claim_full_conformance": False,
        "real_source_documents_imported": False,
    }
    write_json(root / "task-statistics.json", stats)
    if args.plan_only:
        print(json.dumps(stats, indent=2))
        return
    results, pairs, failures = {}, {}, {}

    def persist():
        flat = [p for name in sorted(pairs) for p in pairs[name]]
        status = (
            "running"
            if len(results) + len(failures) < len(plan["shards"])
            else "failed"
            if failures
            else "complete"
        )
        write_json(
            primary / "batch-summary.json",
            {
                "status": status,
                "results": list(results.values()),
                "failed_shards": failures,
            },
        )
        write_json(
            paired / "summary.json",
            {
                **stats,
                "status": status,
                "completed_pairs": len(flat),
                "failed_shards": failures,
                "profile_counts": {"iso": len(flat), "gost": len(flat)},
            },
        )
        temp = paired / "pairs.tmp"
        temp.write_text("".join(json.dumps(p) + "\n" for p in flat))
        temp.replace(paired / "pairs.jsonl")

    def generate(shard):
        result = primary_shard(primary, shard)
        _, rows = complete_shard(primary, paired, shard, 180)
        return result, rows

    persist()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate, s): s["name"] for s in plan["shards"]}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name], pairs[name] = future.result()
                print(name, "COMPLETE", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve every failure, never silently replace tasks
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(name, failures[name], flush=True)
            persist()
    if failures:
        raise RuntimeError("Native generation failed; see summary.json")
    print("All industrial analogue pairs verified", flush=True)


if __name__ == "__main__":
    main()
