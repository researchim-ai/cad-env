"""Generate reproducible, resumable ISO/GOST draft drawing corpora in audited shards."""

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cad_env.learning.audit import audit_dataset
from cad_env.learning.extreme_tasks import FAMILIES, TIERS, make_extreme_task
from cad_env.learning.factory import generate_dataset

from .dataset import audit_technical_dataset, convert_dataset
from .export import sha

VERSION = "cad-env.standard-drawing-batch.v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def select_plan(count=500, shard_size=25, seed_start=9000000, excluded_groups=()):
    if count < 10 or count % 10 or shard_size < 1:
        raise ValueError("Count must be a positive multiple of 10; shard size positive")
    seen = set(excluded_groups)
    combinations = [(f, t) for f in FAMILIES for t in TIERS]
    shards = []
    for standard_index, standard in enumerate(("iso", "gost")):
        selected = []
        targets = {"train": count * 4 // 10, "validation": count // 10}
        for index, (family, tier) in enumerate(combinations):
            remaining = {s: n // 6 + (index < n % 6) for s, n in targets.items()}
            start = seed_start + standard_index * 1000000
            for seed in range(start, start + 100000):
                task = make_extreme_task(family, seed, tier, "image")
                split, group = task["split"], task["split_group"]
                if remaining.get(split, 0) and group not in seen:
                    selected.append(task)
                    seen.add(group)
                    remaining[split] -= 1
                if not any(remaining.values()):
                    break
            if any(remaining.values()):
                raise ValueError("Insufficient distinct naturally split geometries")
        random.Random(f"{VERSION}:{standard}:{seed_start}").shuffle(selected)
        for offset in range(0, len(selected), shard_size):
            shards.append(
                {
                    "name": f"{standard}-{offset // shard_size:03d}",
                    "standard": standard,
                    "tasks": selected[offset : offset + shard_size],
                }
            )
    return {
        "version": VERSION,
        "count": count,
        "seed_start": seed_start,
        "shard_size": shard_size,
        "excluded_groups": sorted(set(excluded_groups)),
        "scope": "Independent synthetic geometries, image-only instructions, draft ISO/GOST profiles; no full standards claim or TEST export",
        "shards": shards,
    }


def quarantine(root, path):
    if path.exists():
        target = (
            root
            / "failed-attempts"
            / (path.parent.name + "-" + path.name + "-" + str(time.time_ns()))
        )
        target.parent.mkdir(exist_ok=True)
        path.rename(target)


def verify_tasks(source, tasks):
    entries = [
        json.loads(s) for s in (source / "manifest.jsonl").read_text().splitlines()
    ]
    if len(entries) != len(tasks) or not all(e["accepted"] for e in entries):
        raise ValueError("Not all prespecified native tasks were accepted")
    expected = {t["task_id"]: t for t in tasks}
    if {e["task_id"] for e in entries} != set(expected):
        raise ValueError("Shard task identities differ from frozen plan")
    for entry in entries:
        task = json.loads(
            (source / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        task.pop("goal_images", None)
        if task != json.loads(json.dumps(expected[entry["task_id"]])):
            raise ValueError("Shard task specification differs from frozen plan")


def run_shard(root, shard, native_workers, timeout):
    name, standard, tasks = shard["name"], shard["standard"], shard["tasks"]
    folder = root / "shards" / name
    folder.mkdir(parents=True, exist_ok=True)
    source, output = folder / "source", folder / "dataset"
    if source.exists():
        try:
            verify_tasks(source, tasks)
            if not audit_dataset(source)["ok"]:
                raise ValueError("Existing native source failed audit")
        except (ValueError, KeyError, OSError):
            quarantine(root, output)
            quarantine(root, source)
    if not source.exists():
        print(f"[{name}] generating {len(tasks)} native tasks", flush=True)
        generate_dataset(
            source,
            task_specs=tasks,
            families=FAMILIES,
            modes=("create",),
            workers=native_workers,
            timeout=timeout,
            preferences=False,
        )
    verify_tasks(source, tasks)
    if not audit_dataset(source)["ok"]:
        raise ValueError("Native source failed audit")
    if output.exists():
        try:
            summary = json.loads((output / "summary.json").read_text())
            if (
                summary["drawing_standard"] != standard
                or not audit_technical_dataset(output)["ok"]
            ):
                raise ValueError("Existing drawing shard failed audit")
        except (ValueError, KeyError, OSError):
            quarantine(root, output)
    if not output.exists():
        print(f"[{name}] exporting {standard.upper()} drawings", flush=True)
        summary = convert_dataset(
            source,
            output,
            standard=standard,
            metadata={
                "synthetic_example": True,
                "owner": "CAD-ENV",
                "status": "DRAFT",
                "issue_date": "2026-11-09",
                "document_type": "Part drawing"
                if standard == "iso"
                else "Чертёж детали",
            },
        )
    if summary["accepted"] != len(tasks) or not audit_technical_dataset(output)["ok"]:
        raise ValueError("Drawing shard is incomplete or failed audit")
    pages, images = 0, 0
    for bundle in (output / "drawings").iterdir():
        report = json.loads((bundle / "conformance.json").read_text())
        if not report["format_checks_passed"] or report["can_claim_full_conformance"]:
            raise ValueError("Draft formatting check failed or false conformance claim")
        manifest = json.loads((bundle / "manifest.json").read_text())
        pages += manifest["page_count"]
        images += len(manifest["vlm_tiles"])
    result = {
        "name": name,
        "standard": standard,
        "status": "complete",
        "dataset": str(output.relative_to(root)),
        "tasks": len(tasks),
        "split_counts": dict(Counter(t["split"] for t in tasks)),
        "feedback_rows": summary["counts"],
        "drawing_bundles": summary["drawing_bundles"],
        "pages": pages,
        "vlm_images": images,
        "format_checks_passed": True,
        "can_claim_full_conformance": False,
        "summary_sha256": sha(output / "summary.json"),
    }
    write_json(folder / "result.json", result)
    print(f"[{name}] COMPLETE: {len(tasks)} tasks, {pages} pages", flush=True)
    return result


def run_batch(
    output,
    *,
    count=500,
    shard_size=25,
    seed_start=9000000,
    jobs=4,
    native_workers=2,
    timeout=180,
    excluded_groups=(),
):
    if jobs < 1 or native_workers < 1:
        raise ValueError("Worker counts must be positive")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = select_plan(count, shard_size, seed_start, excluded_groups)
    encoded = json.dumps(plan, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    plan_path = root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("Existing batch plan differs; use a new output directory")
    elif any(root.iterdir()):
        raise ValueError("Nonempty output without frozen plan")
    else:
        write_json(plan_path, plan)
        snapshot = root / "runner_snapshot"
        shutil.copytree(
            Path(__file__).parent,
            snapshot,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    frozen_path = root / "runner-sources.json"
    current = {
        str(p.relative_to(root / "runner_snapshot")): sha(p)
        for p in (root / "runner_snapshot").rglob("*")
        if p.is_file()
    }
    # Compare the working implementation with the snapshot before resuming.
    for name, expected in current.items():
        actual = Path(__file__).parent / name
        if not actual.is_file() or sha(actual) != expected:
            raise ValueError(
                "Drawing implementation changed since batch creation: " + name
            )
    if not frozen_path.exists():
        write_json(frozen_path, current)
    results, failures = {}, {}
    started = time.monotonic()

    def persist():
        tasks = [t for s in plan["shards"] for t in s["tasks"]]
        result = {
            "version": VERSION,
            "status": "running",
            "planned_tasks": count,
            "plan_sha256": digest,
            "independent_geometries": len({t["split_group"] for t in tasks}),
            "planned_split_counts": dict(Counter(t["split"] for t in tasks)),
            "planned_standard_counts": dict(
                Counter(s["standard"] for s in plan["shards"] for _ in s["tasks"])
            ),
            "completed_tasks": sum(r["tasks"] for r in results.values()),
            "completed_shards": len(results),
            "failed_shards": failures,
            "results": [results[k] for k in sorted(results)],
            "training_datasets": [results[k]["dataset"] for k in sorted(results)],
            "can_claim_full_conformance": False,
            "test_opened": False,
            "training_started": False,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
        }
        if len(results) + len(failures) == len(plan["shards"]):
            result["status"] = "failed" if failures else "complete"
        write_json(root / "batch-summary.json", result)
        return result

    persist()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_shard, root, s, native_workers, timeout): s
            for s in plan["shards"]
        }
        for future in as_completed(futures):
            name = futures[future]["name"]
            try:
                results[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - persist every shard failure, never report partial success
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"[{name}] FAILED: {failures[name]}", flush=True)
            persist()
    result = persist()
    if failures:
        raise RuntimeError(
            "Some prespecified shards failed; inspect batch-summary.json and resume with the same command"
        )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=9000000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--native-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--exclude-dataset", action="append", default=[])
    args = parser.parse_args()
    excluded = set()
    for dataset in args.exclude_dataset:
        for line in (Path(dataset) / "manifest.jsonl").read_text().splitlines():
            excluded.add(json.loads(line)["split_group"])
    result = run_batch(
        args.output,
        count=args.count,
        shard_size=args.shard_size,
        seed_start=args.seed_start,
        jobs=args.jobs,
        native_workers=args.native_workers,
        timeout=args.timeout,
        excluded_groups=excluded,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "completed_tasks", "completed_shards")}
        )
    )


if __name__ == "__main__":
    main()
