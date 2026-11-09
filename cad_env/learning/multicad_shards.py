"""Parallel CPU generation with audited, lineage-preserving shard merging."""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cad_env.multicad.dataset import audit, sha, write_jsonl


def merge(shards, output):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh merged corpus")
    root.mkdir(parents=True)
    chosen = {}
    exports = {s: [] for s in ["train", "validation", "test"]}
    provenance = []
    first_summary = None
    source_hashes = None
    for directory in map(Path, shards):
        checked = audit(directory)
        summary = json.loads((directory / "summary.json").read_text())
        if not checked["ok"] or summary["rejected"]:
            raise ValueError("Incomplete or rejected native shard")
        snapshot = {
            str(p.relative_to(directory / "source_snapshot")): sha(p)
            for p in (directory / "source_snapshot").rglob("*.py")
        }
        if source_hashes is not None and snapshot != source_hashes:
            raise ValueError("Shard source snapshots differ")
        source_hashes = snapshot
        if first_summary is None:
            first_summary = summary
            shutil.copytree(directory / "source_snapshot", root / "source_snapshot")
        elif summary["engines"] != first_summary["engines"]:
            raise ValueError("Shard engine sets differ")
        selected = set()
        for line in (directory / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            if not row["accepted"]:
                raise ValueError("Rejected shard task")
            task = json.loads(
                (directory / "episodes" / row["task_id"] / "task.json").read_text()
            )
            if any(
                task[k] != row[k]
                for k in [
                    "engine",
                    "family",
                    "mode",
                    "input_modality",
                    "split",
                    "split_group",
                ]
            ):
                raise ValueError("Manifest/task disagreement")
            if row["task_id"] in chosen:
                old = chosen[row["task_id"]]
                if any(
                    old[k] != row[k]
                    for k in [
                        "engine",
                        "family",
                        "mode",
                        "input_modality",
                        "split",
                        "split_group",
                    ]
                ):
                    raise ValueError("Conflicting duplicate task")
                continue
            chosen[row["task_id"]] = row
            selected.add(row["task_id"])
            shutil.copytree(
                directory / "episodes" / row["task_id"],
                root / "episodes" / row["task_id"],
                ignore=shutil.ignore_patterns(".runtime"),
            )
        for split, split_rows in exports.items():
            split_rows.extend(
                json.loads(line)
                for line in (directory / "exports" / f"{split}.feedback.jsonl")
                .read_text()
                .splitlines()
                if json.loads(line)["task_id"] in selected
            )
        provenance.append(
            {
                "root": str(directory.resolve()),
                "summary_sha256": sha(directory / "summary.json"),
                "accepted": summary["accepted"],
            }
        )
    rows = list(chosen.values())
    write_jsonl(root / "manifest.jsonl", rows)
    for split, data in exports.items():
        write_jsonl(root / "exports" / f"{split}.feedback.jsonl", data)
    result = {
        **{
            k: v
            for k, v in first_summary.items()
            if k not in ["files", "accepted", "split_counts", "feedback_counts"]
        },
        "accepted": len(rows),
        "source_shards": provenance,
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in exports},
        "feedback_counts": {s: len(data) for s, data in exports.items()},
        "files": {
            str(p.relative_to(root)): sha(p)
            for p in root.rglob("*")
            if p.is_file()
            and ".runtime" not in p.parts
            and p.suffix in {".json", ".jsonl", ".png", ".py", ".FCStd", ".step"}
        },
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    checked = audit(root)
    if not checked["ok"]:
        raise ValueError("Merged corpus audit failed: " + str(checked["errors"][:3]))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--engine", required=True, choices=["freecad", "cadquery"])
    p.add_argument("--output", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--python", required=True)

    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--seeds-per-shard", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if min(args.seeds, args.seeds_per_shard, args.workers) < 1:
        raise ValueError("Positive shard budgets required")
    directory = Path(args.shards_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for start in range(0, args.seeds, args.seeds_per_shard):
        jobs.append(
            (
                start,
                min(args.seeds_per_shard, args.seeds - start),
                directory / f"seeds-{start:04d}",
            )
        )
    state = {"status": "generating", "jobs": []}
    state_path = directory / "status.json"

    def generate(job):
        start, count, target = job
        with (directory / f"{start:04d}.log").open("w") as log:
            subprocess.run(
                [
                    args.python,
                    "-m",
                    "cad_env.multicad",
                    "generate",
                    "--engines",
                    args.engine,
                    "--start-seed",
                    str(start),
                    "--seeds",
                    str(count),
                    "--modes",
                    "create",
                    "edit",
                    "--modalities",
                    "image",
                    "--varied-edits",
                    "--output",
                    str(target),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return {"start_seed": start, "root": str(target), "status": "complete"}

    try:
        state_path.write_text(json.dumps(state, indent=2))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for report in pool.map(generate, jobs):
                state["jobs"].append(report)
                state_path.write_text(json.dumps(state, indent=2))
        state["status"] = "merging"
        state_path.write_text(json.dumps(state, indent=2))
        result = merge([job[2] for job in jobs], args.output)
        state.update(status="complete", accepted=result["accepted"], output=args.output)
        state_path.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        state_path.write_text(json.dumps(state, indent=2))
        raise


if __name__ == "__main__":
    main()
