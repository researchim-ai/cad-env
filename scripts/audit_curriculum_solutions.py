"""Read-only audit, full native replay and independent saved-artifact checks."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.observations import agent_messages, transition
from cad_env.drawings.batch import write_json
from cad_env.learning.policy import checkpoint_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recheck_traces(root, out):
    """Reconstruct prompts using raw actions; reuse completed native checks only.

    Keep the first pass intact, including its auditor serialization discrepancy.
    No benchmark source result or native evaluation is modified.
    """
    path = out / "status.json"
    report = json.loads(path.read_text())
    if report["status"] == "running" or len(report["cases"]) != report["planned"]:
        raise ValueError("A completed full native pass is required")
    shutil.copyfile(path, out / "first-pass.json")
    native_worker = Path(__file__).with_name("verify_saved_cad_worker.py")
    archived_worker = out / "source_snapshot" / native_worker.name
    if ast.dump(ast.parse(native_worker.read_text())) != ast.dump(
        ast.parse(archived_worker.read_text())
    ):
        raise ValueError("Independent native verifier changed since first pass")
    by_run = {}
    all_metrics = {}
    steps = 0
    for run in ("baseline", "checkpoint250"):
        status = json.loads((root / run / "status.json").read_text())
        model = next(iter(status["models"].values()))
        by_run[run] = (
            {(r["task_id"], r["standard"]): r for r in model["episodes"]},
            {
                p["task_id"]: p
                for p in json.loads((root / run / "selection.json").read_text())
            },
        )
        if checkpoint_identity(model["path"]) != model["identity"]:
            report["global_errors"].append(
                "Changed model identity during audit: " + run
            )
        metrics = {}
        for level in ("easy", "medium", "hard"):
            metrics[level] = {}
            for standard in ("iso", "gost"):
                rows = [
                    r
                    for r in model["episodes"]
                    if r["difficulty"] == level and r["standard"] == standard
                ]
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
        if json.loads((root / run / "levels.json").read_text())["metrics"] != metrics:
            report["global_errors"].append("Per-level aggregate mismatch: " + run)
        all_metrics[run] = metrics
    for case in report["cases"]:
        rows, pairs = by_run[case["run"]]
        row = rows[case["task_id"], case["standard"]]
        pair = pairs[case["task_id"]]
        task = json.loads(
            Path(pair["representations"][case["standard"]]["task"]).read_text()
        )
        trace = audit_trace(row, task, pair)
        errors = [
            e for e in case["errors"] if e != "Public prompt provenance mismatch"
        ] + trace["errors"]
        if any(
            row[key] != pair[key]
            for key in ("task_id", "split_group", "family", "difficulty")
        ):
            errors.append("Selection/episode metadata mismatch")
        case["errors"] = sorted(set(errors))
        steps += row["steps"]
        case["prompt_hash_reconstruction"] = (
            "raw model JSON, preserving integer/float serialization"
        )
        write_json(
            out / case["run"] / case["standard"] / case["task_id"] / "audit.json", case
        )
    comparison = json.loads((root / "comparison-checkpoint250.json").read_text())
    if (
        comparison["before"] != all_metrics["baseline"]
        or comparison["after"] != all_metrics["checkpoint250"]
    ):
        report["global_errors"].append("Published comparison aggregate mismatch")
    report["steps_verified"] = steps
    report["native_verifier_sha256"] = sha(archived_worker)
    report["native_verifier_source"] = str(archived_worker)
    report["trace_verifier_sha256"] = sha(__file__)
    report["trace_recheck_note"] = (
        "First pass reconstructed history from normalized execution actions. This pass uses raw model JSON as the original policy did. Completed independent native checks are retained; original benchmark artifacts are unchanged."
    )
    report["confirmed_metrics"] = all_metrics
    report["error_counts"] = dict(
        Counter(e for c in report["cases"] for e in c["errors"])
    )
    report["status"] = (
        "passed"
        if not report["global_errors"] and not report["error_counts"]
        else "discrepancies"
    )
    write_json(path, report)
    print(report["status"], report["error_counts"], "steps", steps, flush=True)


def audit_trace(row, task, pair):
    episode = Path(row["episode_dir"])
    events = [
        json.loads(s) for s in (episode / "trajectory.jsonl").read_text().splitlines()
    ]
    case = episode.parent.parent
    responses = [
        json.loads(s) for s in (case / "responses.jsonl").read_text().splitlines()
    ]
    errors = []
    history = []
    committed = []
    expected_images = [
        sha(p) for p in pair["representations"][row["standard"]]["vlm_images"]
    ]
    if len(events) != row["steps"] or len(responses) != len(events):
        errors.append("Trace/response count mismatch")
    if json.loads((case / "result.json").read_text()) != row:
        errors.append("Per-case result differs from aggregate")
    for event, response in zip(events, responses):
        prompt = agent_messages(task, event["observation_before"], history, visual=True)
        digest = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
        if digest != response["prompt_sha256"]:
            errors.append("Public prompt provenance mismatch")
        if response["image_sha256"] != expected_images:
            errors.append("Goal crop provenance mismatch")
        try:
            action = json.loads(response["response"])
        except ValueError:
            action = {"op": "invalid_json"}
        if action != event["action"]:
            errors.append("Model response/action mismatch")
        if event["info"]["execution_ok"]:
            committed.append(event["action"])
        history.append(
            {
                **transition(action, event["info"], event["observation_after"]),
                "response": response["response"][:2000],
            }
        )
    if committed != row["committed_actions"]:
        errors.append("Committed actions differ from successful native transitions")
    if sum(not e["info"]["execution_ok"] for e in events) != row["invalid_actions"]:
        errors.append("Invalid action count mismatch")
    actual_finish = bool(committed and committed[-1]["op"] == "finish")
    if actual_finish != (row["stop_reason"] == "agent_finish"):
        errors.append("Actual finish differs from recorded stop")
    if bool(row["task_success"]) != (
        actual_finish and bool(row["validation"]["task_success"])
    ):
        errors.append("Success aggregation mismatch")
    if row["stop_reason"] == "agent_finish":
        grade = Path(events[-1]["info"]["result_dir"])
    else:
        grade = case / "final-score"
    native = json.loads((grade / "result.json").read_text())
    if native.get("artifact_sha256") != sha(grade / "output.FCStd"):
        errors.append("Saved FCStd hash mismatch")
    for name, key in [("schema.py", "schema_sha256"), ("worker.py", "worker_sha256")]:
        if native.get(key) != sha(Path("cad_env/agent") / name):
            errors.append("Frozen native source mismatch")
    request = json.loads((grade / "request.json").read_text())
    expected = committed if actual_finish else committed + [{"op": "finish"}]
    if (
        request["actions"] != expected
        or request["reference_actions"] != task["reference_actions"]
    ):
        errors.append("Original scoring request differs from student or reference")
    return {
        "errors": sorted(set(errors)),
        "saved": str(grade / "output.FCStd"),
        "actual_finish": actual_finish,
        "runtime": native.get("runtime"),
        "case": str(case),
    }


def verify(root, run, label, row, pair, out):
    task = json.loads(
        Path(pair["representations"][row["standard"]]["task"]).read_text()
    )
    trace = audit_trace(row, task, pair)
    case = out / run / row["standard"] / row["task_id"]
    case.mkdir(parents=True)
    backend = FreeCADBackend(timeout=180)
    replay = backend.run(
        row["committed_actions"],
        case / "replay",
        initial_actions=task["initial_actions"],
        reference_actions=task["reference_actions"],
        tolerance=task["tolerance"],
        expected_metrics=task["analytic_metrics"],
        reward_mode="strict_success",
    )
    errors = trace["errors"][:]
    if not replay.get("execution_ok") or replay.get("validation", {}).get(
        "evaluator_error"
    ):
        errors.append("Native replay execution/evaluation failure")
    if replay.get("runtime") != trace["runtime"]:
        errors.append("Runtime mismatch")
    success = trace["actual_finish"] and bool(
        replay.get("validation", {}).get("task_success")
    )
    if success != row["task_success"]:
        errors.append("Replayed success differs from report")
    oracle = (
        Path(pair["representations"][row["standard"]]["dataset"])
        / "episodes"
        / row["task_id"]
        / "oracle/output.FCStd"
    )
    req = {
        "reference": str(oracle),
        "saved": trace["saved"],
        "replayed": str(case / "replay/output.FCStd"),
        "drawing": pair["representations"][row["standard"]]["native_model"],
        "analytic": task["analytic_metrics"],
        "tolerance": task["tolerance"],
        "actual_finish": trace["actual_finish"],
        "output": str(case / "independent.json"),
    }
    request = case / "independent-request.json"
    request.write_text(json.dumps(req))
    script = Path(__file__).with_name("verify_saved_cad_worker.py").absolute()
    bootstrap = case / "run-independent.py"
    bootstrap.write_text(
        'import runpy,sys\nsys.argv=["verify",'
        + repr(str(request))
        + "]\nrunpy.run_path("
        + repr(str(script))
        + ',run_name="__main__")\n'
    )
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    for k in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        env[k] = str(case / ".runtime" / k)
    with (case / "independent.log").open("w") as log:
        subprocess.run(
            [backend.executable, str(bootstrap)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    independent = json.loads((case / "independent.json").read_text())
    if not independent["ok"]:
        errors.append(
            "Independent artifact check failed: " + str(independent.get("error"))
        )
    else:
        if not independent["drawing"].get("exact") or not independent["reference"].get(
            "spec_pass"
        ):
            errors.append("Drawing/oracle/analytic reference mismatch")
        for key in ("saved", "replayed"):
            if independent[key]["independent_success"] != row["task_success"]:
                errors.append("Independent " + key + " success differs from report")
    result = {
        "run": run,
        "label": label,
        "task_id": row["task_id"],
        "standard": row["standard"],
        "difficulty": row["difficulty"],
        "recorded_success": row["task_success"],
        "replayed_success": success,
        "actual_finish": trace["actual_finish"],
        "errors": errors,
        "independent": independent,
    }
    write_json(case / "audit.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--recheck-traces", action="store_true")
    args = p.parse_args()
    root = args.root.absolute()
    out = args.output.absolute()
    if args.recheck_traces:
        recheck_traces(root, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    statuses = {}
    global_errors = []
    for run in ("baseline", "checkpoint250"):
        status = json.loads((root / run / "status.json").read_text())
        statuses[run] = status
        if status["status"] != "complete":
            raise ValueError("Complete source benchmark required")
        selected = json.loads((root / run / "selection.json").read_text())
        pairs = {x["task_id"]: x for x in selected}
        expected = {
            (x["task_id"], standard) for x in selected for standard in ("iso", "gost")
        }
        for label, model in status["models"].items():
            rows = model["episodes"]
            keys = [(r["task_id"], r["standard"]) for r in rows]
            if len(set(keys)) != len(keys) or set(keys) != expected:
                global_errors.append(run + ": duplicate/missing cases")
            if checkpoint_identity(model["path"]) != model["identity"]:
                global_errors.append(run + ": checkpoint identity changed")
            jobs.extend((run, label, row, pairs[row["task_id"]]) for row in rows)
    if statuses["baseline"]["protocol"] != statuses["checkpoint250"]["protocol"]:
        global_errors.append("Baseline/candidate protocols differ")
    report = {
        "status": "running",
        "global_errors": global_errors,
        "planned": len(jobs),
        "cases": [],
        "scope": "Full CPU replay plus independently loaded saved artifacts; no new model inference or score replacement.",
    }
    write_json(out / "status.json", report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(verify, root, run, label, row, pair, out): (run, row)
            for run, label, row, pair in jobs
        }
        for f in as_completed(futures):
            run, row = futures[f]
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001 - keep audit failures separate from model scores
                result = {
                    "run": run,
                    "task_id": row["task_id"],
                    "standard": row["standard"],
                    "errors": [repr(exc)],
                }
            report["cases"].append(result)
            write_json(out / "status.json", report)
            if len(report["cases"]) % 20 == 0:
                print(
                    "verified",
                    len(report["cases"]),
                    "/",
                    len(jobs),
                    "discrepancies",
                    sum(bool(c["errors"]) for c in report["cases"]),
                    flush=True,
                )
    report["status"] = (
        "passed"
        if not global_errors and all(not c["errors"] for c in report["cases"])
        else "discrepancies"
    )
    report["error_counts"] = dict(
        Counter(e for c in report["cases"] for e in c["errors"])
    )
    write_json(out / "status.json", report)
    print(report["status"], report["error_counts"], flush=True)


if __name__ == "__main__":
    main()
