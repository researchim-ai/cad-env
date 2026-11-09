"""Run the frozen generated curriculum independently of its generation process."""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad_env.drawings.batch import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="artifacts/models/qwen35-4b")
    parser.add_argument("--label", default="qwen35-base")
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Use a simple run directory name")
    output = root / args.run_name
    if output.exists():
        raise FileExistsError("Preserve existing evaluation; choose a fresh run name")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU1 only")
    if Path(sys.prefix).name != ".venv-qwen35":
        raise ValueError(
            "Use .venv-qwen35/bin/python, preserving the interpreter symlink"
        )
    source = root / "paired"
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Generation must be complete")
    pairs = [json.loads(s) for s in (source / "pairs.jsonl").read_text().splitlines()]
    if any(p["split"] != "validation" for p in pairs):
        raise ValueError("Validation only")
    expected = Counter(p["difficulty"] for p in pairs)
    by_id = {p["task_id"]: p["difficulty"] for p in pairs}
    command = [
        sys.executable,
        "-u",
        "scripts/benchmark_paired_drawings.py",
        "--paired",
        str(source),
        "--output",
        str(output),
        "--model",
        args.label + "=" + str(Path(args.model).resolve()),
        "--count",
        str(len(pairs)),
        "--batch-size",
        "2",
        "--workers",
        "4",
        "--max-new-tokens",
        "1024",
        "--stagnant-limit",
        "6",
    ]
    queue = root / (args.run_name + "-queue.json")
    state = {"status": "running", "command": command, "python_prefix": sys.prefix}
    write_json(queue, state)
    try:
        with (root / (args.run_name + ".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        status = json.loads((output / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError("Benchmark incomplete or infrastructure failures")
        rows = status["models"][args.label]["episodes"]
        metrics = {}
        for level, count in expected.items():
            metrics[level] = {}
            for standard in ("iso", "gost"):
                group = [
                    r
                    for r in rows
                    if r["standard"] == standard and by_id[r["task_id"]] == level
                ]
                if len(group) != count or any(r["status"] != "complete" for r in group):
                    raise ValueError("Missing prespecified episodes")
                metrics[level][standard] = {
                    "evaluated": len(group),
                    "successes": sum(r["task_success"] for r in group),
                    "mean_geometry_score": sum(
                        r["validation"]["geometry_score"] for r in group
                    )
                    / len(group),
                    "actions": sum(r["steps"] for r in group),
                    "invalid_actions": sum(r["invalid_actions"] for r in group),
                }
        write_json(output / "levels.json", {"status": "complete", "metrics": metrics})
        baseline = root / "baseline"
        if output != baseline and (baseline / "status.json").exists():
            before = json.loads((baseline / "status.json").read_text())
            if (
                before["status"] != "complete"
                or before["protocol"] != status["protocol"]
            ):
                raise ValueError(
                    "Baseline/candidate protocol or task selection differs"
                )
            previous = json.loads((baseline / "levels.json").read_text())
            write_json(
                root / ("comparison-" + args.run_name + ".json"),
                {
                    "status": "complete",
                    "before": previous["metrics"],
                    "after": metrics,
                    "model": str(Path(args.model).resolve()),
                    "same_selection_sha256": status["protocol"]["selection_sha256"],
                    "scope": "Same 90 development-validation parents, paired profiles; not final TEST.",
                },
            )
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(queue, state)


if __name__ == "__main__":
    main()
