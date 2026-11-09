"""Native audited phase subsets for a matched-data control experiment."""

import argparse
import json
import random
import shutil
from pathlib import Path

from .audit import audit_dataset
from .perception_data import sha
from .visual_audit import audit_visual


def selected_rows(source, ids, phase):
    rows = [
        r
        for r in map(
            json.loads,
            (source / "exports/train.feedback.jsonl").read_text().splitlines(),
        )
        if r["task_id"] in ids
    ]
    if phase == "revisit_general":
        reserved = {
            (r["task_id"], r["step_id"])
            for r in selected_rows(source, ids, "stop_contrast")
        }
        return [r for r in rows if (r["task_id"], r["step_id"]) not in reserved]
    if phase != "stop_contrast":
        return rows
    tasks = {
        task_id: json.loads((source / "episodes" / task_id / "task.json").read_text())
        for task_id in ids
    }
    first = [
        r
        for r in rows
        if r["step_id"] == 0
        and tasks[r["task_id"]]["curriculum_state"] == "complete_peer_with_repeats"
    ]
    late = [
        r
        for r in rows
        if r["step_id"] > 0
        and json.loads(r["completion"][0]["content"][0]["text"])["op"] == "finish"
    ]
    random.Random(2026).shuffle(late)
    return first + late[:50]


def build(source, output, phase):
    source, out = Path(source).resolve(), Path(output).resolve()
    base = audit_visual(source)
    if not base["ok"]:
        raise ValueError(base["errors"])
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    hashes = {}
    for e in map(json.loads, (source / "manifest.jsonl").read_text().splitlines()):
        folder = source / "episodes" / e["task_id"]
        t = json.loads((folder / "task.json").read_text())
        if t["curriculum_phase"] != (
            "revisit" if phase in ("stop_contrast", "revisit_general") else phase
        ):
            continue
        entries.append(e)
        for rel in (
            "task.json",
            "goal.png",
            "oracle/result.json",
            "oracle/output.FCStd",
        ):
            dest = out / "episodes" / e["task_id"] / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / rel, dest)
            hashes[str(dest.relative_to(out))] = sha(dest)
    ids = {e["task_id"] for e in entries}
    rows = selected_rows(source, ids, phase)
    if not entries or not rows:
        raise ValueError("Empty phase subset")
    (out / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    hashes["manifest.jsonl"] = sha(out / "manifest.jsonl")
    (out / "exports").mkdir()
    export = out / "exports/train.feedback.jsonl"
    export.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = json.loads((source / "summary.json").read_text())
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "release-subset-v1",
        "action_contract": "single_action",
        "runtime": original["runtime"],
        "attempted": len(entries),
        "accepted": len(entries),
        "rejected": 0,
        "counts": {"train.feedback": len(rows)},
        "supported_training_exports": ["train.feedback.jsonl"],
        "artifact_sha256": hashes,
        "export_sha256": {"exports/train.feedback.jsonl": sha(export)},
        "phase": phase,
        "source_root": str(source),
        "source_summary_sha256": sha(source / "summary.json"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_subset(out)
    (out / "subset-audit.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise ValueError(report["errors"])
    return report


def audit_subset(root, base_report=None):
    root = Path(root).resolve()
    summary = json.loads((root / "summary.json").read_text())
    source = Path(summary["source_root"])
    base = base_report or audit_dataset(root)
    visual = audit_visual(root, base_report=base)
    errors = list(visual["errors"])
    if sha(source / "summary.json") != summary["source_summary_sha256"]:
        errors.append("Source identity mismatch")
    expected = []
    for e in map(json.loads, (source / "manifest.jsonl").read_text().splitlines()):
        if json.loads((source / "episodes" / e["task_id"] / "task.json").read_text())[
            "curriculum_phase"
        ] == (
            "revisit"
            if summary["phase"] in ("stop_contrast", "revisit_general")
            else summary["phase"]
        ):
            expected.append(e)
    actual = list(map(json.loads, (root / "manifest.jsonl").read_text().splitlines()))
    if actual != expected:
        errors.append("Incorrect phase manifest")
    ids = {e["task_id"] for e in expected}
    expected_rows = selected_rows(source, ids, summary["phase"])
    rows = list(
        map(
            json.loads, (root / "exports/train.feedback.jsonl").read_text().splitlines()
        )
    )
    if rows != expected_rows:
        errors.append("Changed or missing source feedback")
    for e in actual:
        for rel in (
            "task.json",
            "goal.png",
            "oracle/result.json",
            "oracle/output.FCStd",
        ):
            if sha(root / "episodes" / e["task_id"] / rel) != sha(
                source / "episodes" / e["task_id"] / rel
            ):
                errors.append("Changed native phase artifact")
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "rows": len(rows),
        "phase": summary["phase"],
        "train_only_export": True,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--phase",
        required=True,
        choices=["clean", "revisit", "stop_contrast", "revisit_general"],
    )
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.phase), indent=2))


if __name__ == "__main__":
    main()
