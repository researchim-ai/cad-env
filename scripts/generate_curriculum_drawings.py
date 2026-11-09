"""Generate matched ISO/GOST TRAIN or validation corpora at three fixed levels."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.batch import run_shard, write_json
from cad_env.drawings.export import sha
from cad_env.learning.extreme_tasks import FAMILIES, make_extreme_task
from scripts.generate_paired_drawings import complete_shard

LEVELS = {"easy": "easy", "medium": "challenge", "hard": "extreme"}


def select_plan(
    per_level=30,
    levels=tuple(LEVELS),
    split="validation",
    seed=21000000,
    shard_size=10,
    excluded=(),
):
    if per_level < 1 or shard_size < 1 or split not in ("train", "validation"):
        raise ValueError("Positive counts and TRAIN/development-validation only")
    if not levels or len(set(levels)) != len(levels) or set(levels) - LEVELS.keys():
        raise ValueError("Choose distinct easy/medium/hard levels")
    seen, shards = set(excluded), []
    for level in levels:
        tasks = []
        for index, family in enumerate(FAMILIES):
            quota = per_level // len(FAMILIES) + (index < per_level % len(FAMILIES))
            accepted = 0
            for s in range(seed, seed + max(10000, quota * 1000)):
                if accepted == quota:
                    break
                t = make_extreme_task(family, s, LEVELS[level], "image")
                if t["split"] != split or t["split_group"] in seen:
                    continue
                seen.add(t["split_group"])
                t["source_difficulty"] = t["difficulty"]
                t["difficulty"] = level
                t["curriculum_version"] = "three-level-drawings-v1"
                tasks.append(t)
                accepted += 1
            if accepted != quota:
                raise ValueError("Sampling window exhausted; choose a new seed range")
        # Mix families across shards without filtering on model outcomes.
        tasks.sort(key=lambda t: hashlib.sha256(t["task_id"].encode()).hexdigest())
        for offset in range(0, len(tasks), shard_size):
            shards.append(
                {
                    "name": f"{level}-{offset // shard_size:04d}",
                    "standard": "iso" if offset // shard_size % 2 == 0 else "gost",
                    "tasks": tasks[offset : offset + shard_size],
                }
            )
    return json.loads(
        json.dumps(
            {
                "version": "three-level-drawings-v1",
                "count": per_level * len(levels),
                "per_level": per_level,
                "levels": list(levels),
                "split": split,
                "seed": seed,
                "excluded_groups": sorted(set(excluded)),
                "shards": shards,
                "split_policy": "Natural lineage hash; quotas sampled without split reassignment",
                "can_claim_full_conformance": False,
            }
        )
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--per-level", type=int, default=30)
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--seed", type=int, default=21000000)
    p.add_argument("--shard-size", type=int, default=10)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--exclude-pairs", action="append", default=[])
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--benchmark-base", action="store_true")
    p.add_argument("--wait-experiment")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("Positive worker count required")
    if args.benchmark_base and (args.split != "validation" or args.per_level % 3):
        p.error(
            "Baseline benchmark needs validation and equal family quotas (per-level multiple of 3)"
        )
    excluded = set()
    for filename in args.exclude_pairs:
        excluded.update(
            json.loads(line)["split_group"]
            for line in Path(filename).read_text().splitlines()
        )
    plan = select_plan(
        args.per_level, args.levels, args.split, args.seed, args.shard_size, excluded
    )
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    primary, paired = root / "primary", root / "paired"
    primary.mkdir(exist_ok=True)
    paired.mkdir(exist_ok=True)
    frozen = primary / "plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("Frozen plan differs; choose a fresh output directory")
    write_json(frozen, plan)
    sources = [
        Path(__file__),
        Path("scripts/generate_paired_drawings.py"),
        Path("cad_env/learning/extreme_tasks.py"),
        *Path("cad_env/drawings").glob("*.py"),
    ]
    hashes = {str(s.resolve()): sha(s) for s in sources}
    provenance = root / "source-hashes.json"
    if provenance.exists() and json.loads(provenance.read_text()) != hashes:
        raise ValueError("Generation implementation changed; choose a fresh output")
    write_json(provenance, hashes)
    tasks = [t for shard in plan["shards"] for t in shard["tasks"]]
    counts = {
        level: [len(t["oracle_actions"]) for t in tasks if t["difficulty"] == level]
        for level in args.levels
    }
    write_json(
        root / "task-statistics.json",
        {
            "geometries": len(tasks),
            "documents": len(tasks) * 2,
            "levels": {
                level: {"count": len(v), "teacher_action_range": [min(v), max(v)]}
                for level, v in counts.items()
            },
            "split": args.split,
            "training_started": False,
            "can_claim_full_conformance": False,
        },
    )
    if args.plan_only:
        print((root / "task-statistics.json").read_text())
        return
    completed, pairs, failures = {}, {}, {}

    def persist():
        flat = [p for name in sorted(pairs) for p in pairs[name]]
        status = (
            "running"
            if len(completed) + len(failures) < len(plan["shards"])
            else "failed"
            if failures
            else "complete"
        )
        write_json(
            primary / "batch-summary.json",
            {
                "status": status,
                "results": list(completed.values()),
                "failed_shards": failures,
            },
        )
        write_json(
            paired / "summary.json",
            {
                "status": status,
                "completed_pairs": len(flat),
                "profile_counts": {"iso": len(flat), "gost": len(flat)},
                "failed_shards": failures,
                "split_counts_by_geometry": dict(Counter(p["split"] for p in flat)),
                "difficulty_counts": dict(Counter(p["difficulty"] for p in flat)),
                "can_claim_full_conformance": False,
            },
        )
        # Atomic readers must never see a partially written index.
        tmp = paired / "pairs.tmp"
        tmp.write_text("".join(json.dumps(p) + "\n" for p in flat))
        tmp.replace(paired / "pairs.jsonl")

    write_json(
        paired / "request.json",
        {"primary": str(primary), "primary_plan_sha256": sha(frozen)},
    )
    persist()

    def generate(shard):
        result = run_shard(primary, shard, 1, 180)
        _, rows = complete_shard(primary, paired, shard, 180)
        return result, rows

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate, s): s["name"] for s in plan["shards"]}
        for future in as_completed(futures):
            name = futures[future]
            try:
                completed[name], pairs[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - record every failed shard, never report partial success
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(name, failures[name], flush=True)
            persist()
    if failures:
        raise RuntimeError(
            "Generation failed; inspect summary and resume unchanged command"
        )
    if args.benchmark_base:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("Physical GPU1 only")
        queue = root / "baseline-queue.json"
        write_json(
            queue, {"status": "waiting", "wait_experiment": args.wait_experiment}
        )
        if args.wait_experiment:
            while json.loads(Path(args.wait_experiment).read_text())["status"] not in (
                "complete",
                "failed",
                "paused",
            ):
                time.sleep(30)
        # Never share the training GPU with another compute process.
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "1",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if gpu.strip():
            raise RuntimeError("GPU1 still busy; baseline not started")
        command = [
            str(Path(".venv-qwen35/bin/python").absolute()),
            "-u",
            "scripts/benchmark_paired_drawings.py",
            "--paired",
            str(paired),
            "--output",
            str(root / "baseline"),
            "--model",
            "qwen35-base=" + str(Path("artifacts/models/qwen35-4b").resolve()),
            "--count",
            str(plan["count"]),
            "--batch-size",
            "2",
            "--workers",
            "4",
            "--max-new-tokens",
            "1024",
            "--stagnant-limit",
            "6",
        ]
        write_json(queue, {"status": "running", "command": command})
        with (root / "baseline.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        write_json(
            queue,
            {
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
            },
        )
        if result.returncode:
            raise RuntimeError("Baseline failed; see baseline.log")
        status = json.loads((root / "baseline/status.json").read_text())
        if status["status"] != "complete":
            write_json(
                queue,
                {
                    "status": "failed",
                    "reason": "Incomplete benchmark or infrastructure failures",
                },
            )
            raise RuntimeError("Baseline did not complete all episodes")
        model = status["models"]["qwen35-base"]
        level_by_id = {
            p["task_id"]: p["difficulty"] for group in pairs.values() for p in group
        }
        metrics = {}
        for level in args.levels:
            metrics[level] = {}
            for standard in ("iso", "gost"):
                rows = [
                    r
                    for r in model["episodes"]
                    if level_by_id[r["task_id"]] == level and r["standard"] == standard
                ]
                if len(rows) != args.per_level or any(
                    r["status"] != "complete" for r in rows
                ):
                    raise ValueError("Missing prespecified level/profile episode")
                metrics[level][standard] = {
                    "evaluated": len(rows),
                    "successes": sum(r["task_success"] for r in rows),
                    "mean_geometry_score": sum(
                        r["validation"]["geometry_score"] for r in rows
                    )
                    / len(rows),
                    "actions": sum(r["steps"] for r in rows),
                    "invalid_actions": sum(r["invalid_actions"] for r in rows),
                }
        write_json(
            root / "baseline/levels.json",
            {
                "status": "complete",
                "metrics": metrics,
                "scope": "Three fixed procedural levels, development-validation only; draft ISO/GOST profiles",
            },
        )


if __name__ == "__main__":
    main()
