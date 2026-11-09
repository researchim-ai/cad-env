"""Package only causal, native-successful TRAIN student recoveries for SFT."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from cad_env.agent.observations import agent_messages, policy_images, transition

from .audit import audit_dataset
from .factory import teacher_transition
from .perception_data import sha
from .recovery_teacher import repair_actions


def verified_rows(task, record):
    """Reconstruct every teacher input from pre-action native events/history."""
    if task["split"] != "train" or not record["verified"]:
        raise ValueError("Verified TRAIN recovery required")
    trajectory = Path(record["trajectory"])
    if sha(trajectory) != record["trajectory_sha256"]:
        raise ValueError("Changed native trajectory")
    events = list(map(json.loads, trajectory.read_text().splitlines()))
    start = record["student_steps"]
    evidence = record["student_evidence"]
    if len(events) <= start:
        raise ValueError("Missing teacher continuation")
    if not events[-1]["info"].get("validation", {}).get("task_success"):
        raise ValueError("Native terminal failure")
    expected_actions = repair_actions(
        task,
        events[0]["observation_before"]["state"],
        events[start]["observation_before"]["state"],
    )
    if [e["action"] for e in events[start:]] != expected_actions:
        raise ValueError("Incorrect teacher recovery")
    history = []
    rows = []
    for i, event in enumerate(events):
        prompt = agent_messages(task, event["observation_before"], history, visual=True)
        images = policy_images(event["observation_before"])
        if i < start:
            proof = evidence[i]
            if (
                prompt != proof["prompt"]
                or images != proof["images"]
                or proof["observation"] != event["observation_before"]
            ):
                raise ValueError("Student causal input mismatch")
            response = proof["response"]
            try:
                action = json.loads(response)
            except ValueError:
                action = {"op": "invalid_json"}
            if action != event["action"]:
                raise ValueError("Student action provenance mismatch")
            history.append(
                {
                    **transition(action, event["info"], event["observation_after"]),
                    "response": response[:2000],
                }
            )
        else:
            row = {
                "task_id": task["task_id"],
                "split_group": task["split_group"],
                "step_id": i,
                "prompt": prompt,
                "completion": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(expected_actions[i - start]),
                            }
                        ],
                    }
                ],
                "images": images,
            }
            rows.append(row)
            history.append(
                teacher_transition(
                    expected_actions[i - start],
                    event["info"],
                    event["observation_after"],
                )
            )
    if rows != record["candidate_feedback"]:
        raise ValueError("Candidate feedback is not causal native supervision")
    return rows, events[-1]["info"]["result_dir"]


def package_rows(root, task, rows):
    result = copy.deepcopy(rows)
    for row in result:
        if len(row["images"]) != 1:
            raise ValueError("This package supports goal-only observations")
        row["images"] = [f"episodes/{task['task_id']}/goal.png"]
    return result


def build(source, collection, output):
    source, collection, out = (Path(p).resolve() for p in (source, collection, output))
    report = json.loads((collection / "status.json").read_text())
    if report["status"] != "complete" or report.get("current_render"):
        raise ValueError("Completed goal-only collection required")
    if sha(source / "summary.json") != report["source_summary_sha256"]:
        raise ValueError("Collection source mismatch")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = []
    hashes = {}
    for ep in report["episodes"]:
        if not ep["verified"]:
            continue
        record = json.loads((collection / (ep["task_id"] + ".jsonl")).read_text())
        folder = source / "episodes" / ep["task_id"]
        task = json.loads((folder / "task.json").read_text())
        if record["current_render"]:
            raise ValueError("Unexpected current-render protocol")
        feedback, final = verified_rows(task, record)
        dest = out / "episodes" / task["task_id"]
        (dest / "oracle").mkdir(parents=True)
        for src, rel in [
            (folder / "task.json", "task.json"),
            (folder / "goal.png", "goal.png"),
            (Path(final) / "output.FCStd", "oracle/output.FCStd"),
            (Path(final) / "result.json", "oracle/result.json"),
        ]:
            shutil.copyfile(src, dest / rel)
            hashes[str((dest / rel).relative_to(out))] = sha(dest / rel)
        evidence = dest / "recovery-proof.json"
        evidence.write_text(json.dumps(record))
        hashes[str(evidence.relative_to(out))] = sha(evidence)
        rows.extend(package_rows(out, task, feedback))
        manifest.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "mode": "edit",
                "split": "train",
                "split_group": task["split_group"],
                "accepted": True,
                "artifact_sha256": sha(dest / "oracle/output.FCStd"),
            }
        )
    if not rows:
        raise ValueError("No verified recoveries; do not fabricate or repeat rows")
    (out / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in manifest))
    hashes["manifest.jsonl"] = sha(out / "manifest.jsonl")
    (out / "exports").mkdir()
    export = out / "exports/train.feedback.jsonl"
    export.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "policy-recovery-v1",
        "action_contract": "single_action",
        "runtime": json.loads((source / "summary.json").read_text())["runtime"],
        "attempted": len(manifest),
        "accepted": len(manifest),
        "rejected": 0,
        "counts": {"train.feedback": len(rows)},
        "supported_training_exports": ["train.feedback.jsonl"],
        "artifact_sha256": hashes,
        "export_sha256": {"exports/train.feedback.jsonl": sha(export)},
        "source_root": str(source),
        "source_summary_sha256": sha(source / "summary.json"),
        "collection_root": str(collection),
        "collection_status_sha256": sha(collection / "status.json"),
        "model_identity": report["model_identity"],
        "scope": "Verified real student prefixes and causal teacher continuations, TRAIN only; no fabricated policy mistakes",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_recoveries(out)
    (out / "recovery-audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["ok"]:
        raise ValueError(audit["errors"])
    return audit


def audit_recoveries(root, base_report=None):
    root = Path(root).resolve()
    base = base_report or audit_dataset(root)
    errors = list(base["errors"])
    summary = json.loads((root / "summary.json").read_text())
    source = Path(summary["source_root"])
    collection = Path(summary["collection_root"])
    if (
        sha(source / "summary.json") != summary["source_summary_sha256"]
        or sha(collection / "status.json") != summary["collection_status_sha256"]
    ):
        errors.append("Changed source/collection identity")
    expected = []
    for e in map(json.loads, (root / "manifest.jsonl").read_text().splitlines()):
        folder = root / "episodes" / e["task_id"]
        task = json.loads((folder / "task.json").read_text())
        for relative in ("task.json", "goal.png"):
            if sha(folder / relative) != sha(
                source / "episodes" / e["task_id"] / relative
            ):
                errors.append("Changed TRAIN task/goal")
        record = json.loads((folder / "recovery-proof.json").read_text())
        if record != json.loads((collection / (e["task_id"] + ".jsonl")).read_text()):
            errors.append("Recovery proof differs from collected model trajectory")
        try:
            rows, final = verified_rows(task, record)
            expected.extend(package_rows(root, task, rows))
            for rel in ("output.FCStd", "result.json"):
                if sha(folder / "oracle" / rel) != sha(Path(final) / rel):
                    errors.append("Changed native recovery reference")
        except ValueError as exc:
            errors.append(str(exc))
    actual = list(
        map(
            json.loads, (root / "exports/train.feedback.jsonl").read_text().splitlines()
        )
    )
    if actual != expected:
        errors.append("Changed or missing causal recovery rows")
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "rows": len(actual),
        "train_only": True,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("source", "collection", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.collection, a.output), indent=2))


if __name__ == "__main__":
    main()
