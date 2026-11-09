"""Complete both drawing profiles for each geometry in an existing frozen batch.

Does not sample new tasks, reassign splits or run training. The original batch may
still be running; only audited primary shards are consumed.
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.drawings.batch import verify_tasks, write_json
from cad_env.drawings.dataset import audit_technical_dataset, convert_dataset
from cad_env.drawings.export import sha


def opposite(standard):
    if standard not in ("iso", "gost"):
        raise ValueError("Expected explicit iso/gost primary profile")
    return "gost" if standard == "iso" else "iso"


def wait_for_primary(primary, shard, timeout):
    result = primary / "shards" / shard["name"] / "result.json"
    started = time.monotonic()
    while not result.is_file():
        summary = json.loads((primary / "batch-summary.json").read_text())
        if shard["name"] in summary["failed_shards"]:
            raise RuntimeError("Primary shard failed: " + shard["name"])
        if time.monotonic() - started > timeout:
            raise TimeoutError("Primary shard not complete: " + shard["name"])
        time.sleep(2)
    result = json.loads(result.read_text())
    if result["status"] != "complete" or result["tasks"] != len(shard["tasks"]):
        raise ValueError("Incomplete primary shard")
    return result


def complete_shard(primary, output, shard, timeout):
    result = wait_for_primary(primary, shard, timeout)
    name, standard = shard["name"], opposite(shard["standard"])
    source = primary / "shards" / name / "source"
    original = primary / result["dataset"]
    if sha(original / "summary.json") != result["summary_sha256"]:
        raise ValueError("Primary summary changed")
    verify_tasks(source, shard["tasks"])
    converted = output / "shards" / name / "dataset"
    if converted.exists():
        summary = json.loads((converted / "summary.json").read_text())
        if summary["drawing_standard"] != standard:
            raise ValueError("Wrong opposite profile")
    else:
        print(
            f"[{name}] exporting paired {standard.upper()} representation", flush=True
        )
        summary = convert_dataset(
            source,
            converted,
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
    audit = audit_technical_dataset(converted)
    if not audit["ok"] or summary["accepted"] != len(shard["tasks"]):
        raise ValueError(
            "Opposite-profile corpus audit failed: " + str(audit["errors"][:3])
        )
    pairs = []
    for task in shard["tasks"]:
        identity, group = task["task_id"], task["split_group"]
        folders = {shard["standard"]: original, standard: converted}
        item = {
            "task_id": identity,
            "split_group": group,
            "split": task["split"],
            "family": task["family"],
            "difficulty": task["difficulty"],
            "representations": {},
        }
        source_hashes, labels = set(), []
        for profile, dataset in folders.items():
            exported = json.loads(
                (dataset / "episodes" / identity / "task.json").read_text()
            )
            assert (
                exported["split_group"] == group and exported["split"] == task["split"]
            )
            assert exported["drawing_standard"] == profile
            labels.append(exported["oracle_actions"])
            bundle = dataset / exported["drawing_bundle"]
            manifest = json.loads((bundle / "manifest.json").read_text())
            report = json.loads((bundle / "conformance.json").read_text())
            if (
                not report["format_checks_passed"]
                or report["can_claim_full_conformance"]
            ):
                raise ValueError("Formatting failure or false full-conformance claim")
            source_hashes.add(manifest["source_sha256"])
            item["representations"][profile] = {
                "dataset": str(dataset),
                "task": str(dataset / "episodes" / identity / "task.json"),
                "pdf": str(bundle / "drawing.pdf"),
                "native_model": str(bundle / "model-with-drawings.FCStd"),
                "page_count": manifest["page_count"],
                "vlm_images": [str(bundle / p) for p in manifest["vlm_tiles"]],
                "source_sha256": manifest["source_sha256"],
                "format_checks_passed": True,
                "can_claim_full_conformance": False,
            }
        if len(source_hashes) != 1 or not (
            labels[0] == labels[1] == task["oracle_actions"]
        ):
            raise ValueError("Paired profiles changed CAD source or teacher actions")
        pairs.append(item)
    completed = {
        "name": name,
        "status": "complete",
        "pairs": len(pairs),
        "added_profile": standard,
        "added_dataset": str(converted),
        "summary_sha256": sha(converted / "summary.json"),
        "audit": audit,
    }
    write_json(output / "shards" / name / "result.json", completed)
    (output / "shards" / name / "pairs.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs)
    )
    print(
        f"[{name}] paired {len(pairs)} identical geometries in ISO and GOST", flush=True
    )
    return completed, pairs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.jobs < 1 or args.wait_timeout < 1:
        parser.error("Worker count and wait timeout must be positive")
    primary, output = Path(args.primary).resolve(), Path(args.output).resolve()
    plan = json.loads((primary / "plan.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "primary": str(primary),
        "primary_plan_sha256": sha(primary / "plan.json"),
        "script_sha256": sha(__file__),
        "geometry_count": plan["count"],
        "profiles": ["iso", "gost"],
        "new_geometries_sampled": 0,
        "can_claim_full_conformance": False,
    }
    request = output / "request.json"
    if request.exists():
        if json.loads(request.read_text()) != frozen:
            raise ValueError("Paired generation request changed; use a fresh output")
    else:
        write_json(request, frozen)
        (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    completed, failures, pairs = {}, {}, {}
    started = time.monotonic()

    def persist():
        flattened = [p for name in sorted(pairs) for p in pairs[name]]
        summary = {
            **frozen,
            "status": "running",
            "completed_pairs": len(flattened),
            "completed_documents": len(flattened) * 2,
            "failed_shards": failures,
            "profile_counts": {"iso": len(flattened), "gost": len(flattened)},
            "split_counts_by_geometry": dict(Counter(p["split"] for p in flattened)),
            "primary_and_opposite_sources_identical": True,
            "teacher_actions_and_splits_preserved": True,
            "training_started": False,
            "test_opened": False,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "shards": [completed[k] for k in sorted(completed)],
        }
        if len(completed) + len(failures) == len(plan["shards"]):
            summary["status"] = "failed" if failures else "complete"
        write_json(output / "summary.json", summary)
        (output / "pairs.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in flattened)
        )
        return summary

    persist()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(complete_shard, primary, output, s, args.wait_timeout): s[
                "name"
            ]
            for s in plan["shards"]
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                completed[name], pairs[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve all failures, never claim partial success
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"[{name}] FAILED: {failures[name]}", flush=True)
            persist()
    summary = persist()
    if failures:
        raise RuntimeError("Paired generation failed; see summary.json")
    if (
        len({p["split_group"] for rows in pairs.values() for p in rows})
        != plan["count"]
    ):
        raise ValueError("Duplicate or missing physical lineage")
    print(
        json.dumps(
            {k: summary[k] for k in ("status", "completed_pairs", "profile_counts")}
        )
    )


if __name__ == "__main__":
    main()
