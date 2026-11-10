"""Resume an interrupted frozen benchmark, retaining completed episode results."""

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from scripts.benchmark_paired_drawings import (
    BatchedHFPolicy,
    PolicyBatcher,
    evaluate_case,
    sha,
    summarize,
    write_json,
)


def pending_cases(selected, rows):
    expected = {(p["task_id"], s) for p in selected for s in ("iso", "gost")}
    keys = [(r["task_id"], r["standard"]) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Duplicate or unexpected saved episode")
    done = {(r["task_id"], r["standard"]) for r in rows if r["status"] == "complete"}
    return [
        (p, s)
        for p in selected
        for s in ("iso", "gost")
        if (p["task_id"], s) not in done
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="tuned-iso-gost")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU 1 required")
    out = args.output.absolute()
    lock = (out / "resume.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = json.loads((out / "status.json").read_text())
    protocol = status["protocol"]
    if sha(out / "selection.json") != protocol["selection_sha256"]:
        raise ValueError("Selection changed")
    if (
        sha(Path(__file__).parent / "benchmark_paired_drawings.py")
        != protocol["runner_sha256"]
    ):
        raise ValueError("Frozen episode runner changed")
    selected = json.loads((out / "selection.json").read_text())
    model = status["models"][args.label]
    todo = pending_cases(selected, model["episodes"])
    archive = out / ("interrupted-" + str(time.time_ns()))
    archive.mkdir()
    shutil.copyfile(out / "status.json", archive / "status.json")
    for pair, standard in todo:
        case = out / args.label / standard / pair["task_id"]
        if case.exists():
            dest = archive / standard / pair["task_id"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(case), str(dest))
    rows = [r for r in model["episodes"] if r["status"] == "complete"]
    resume = {
        "status": "loading",
        "pid": os.getpid(),
        "retained": len(rows),
        "remaining": len(todo),
        "archive": str(archive),
        "resume_runner_sha256": sha(__file__),
    }
    write_json(out / "resume.json", resume)
    print(f"Retained {len(rows)}; restarting {len(todo)} episodes", flush=True)
    try:
        policy = BatchedHFPolicy(
            model["path"],
            max_context=protocol["max_context"],
            max_pixels=protocol["max_pixels_per_image"],
            temperature=0,
            max_new_tokens=protocol["max_new_tokens_per_action"],
            chat_template_kwargs={"enable_thinking": False},
        )
        if policy.identity != model["identity"]:
            raise ValueError("Checkpoint identity changed")
        batcher = PolicyBatcher(policy, batch_size=protocol["batch_size"])
        resume["status"] = "evaluating"
        write_json(out / "resume.json", resume)
        print("Checkpoint identity verified; evaluating on physical GPU 1", flush=True)
        try:
            with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
                futures = {
                    pool.submit(
                        evaluate_case,
                        p,
                        s,
                        batcher,
                        out / args.label,
                        protocol["stagnation_guard"],
                    ): (p, s)
                    for p, s in todo
                }
                for future in as_completed(futures):
                    pair, standard = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001 - retain infrastructure failures separately
                        row = {
                            "task_id": pair["task_id"],
                            "standard": standard,
                            "status": "infrastructure_error",
                            "error": repr(exc),
                        }
                        print("INFRASTRUCTURE ERROR", row, flush=True)
                    rows.append(row)
                    model.update(
                        episodes=rows, metrics=summarize(rows), status="evaluating"
                    )
                    status["status"] = "evaluating"
                    write_json(out / "status.json", status)
                    resume.update(completed=len(rows), last_result_unix=time.time())
                    write_json(out / "resume.json", resume)
        finally:
            batcher.close()
        final = "complete" if all(r["status"] == "complete" for r in rows) else "failed"
        model["status"] = status["status"] = final
        write_json(out / "status.json", status)
        write_json(
            out / args.label / "evaluation.json",
            {
                "status": final,
                "model_identity": policy.identity,
                "protocol": protocol,
                "episodes": rows,
                "metrics": summarize(rows),
                "resumed": True,
                "resumed_segment_batch_stats": batcher.stats,
            },
        )
        if final == "complete":
            metrics = {}
            for level in sorted({p["difficulty"] for p in selected}):
                metrics[level] = {}
                ids = {p["task_id"] for p in selected if p["difficulty"] == level}
                for standard in ("iso", "gost"):
                    group = [
                        r
                        for r in rows
                        if r["task_id"] in ids and r["standard"] == standard
                    ]
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
            write_json(out / "levels.json", {"status": final, "metrics": metrics})
            before = json.loads((out.parent / "baseline/status.json").read_text())
            if before["status"] != "complete" or before["protocol"] != protocol:
                raise ValueError("Baseline protocol differs")
            write_json(
                out.parent / ("comparison-" + out.name + ".json"),
                {
                    "status": final,
                    "before": json.loads(
                        (out.parent / "baseline/levels.json").read_text()
                    )["metrics"],
                    "after": metrics,
                    "model": model["path"],
                    "same_selection_sha256": protocol["selection_sha256"],
                    "scope": "Same development-validation parents; interrupted episodes restarted; not final TEST.",
                },
            )
        resume["status"] = final
    except BaseException as exc:
        resume.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(out / "resume.json", resume)
        queue = out.parent / (out.name + "-queue.json")
        old = json.loads(queue.read_text()) if queue.exists() else {}
        old.update(status=resume["status"], resume=str(out / "resume.json"))
        write_json(queue, old)


if __name__ == "__main__":
    main()
