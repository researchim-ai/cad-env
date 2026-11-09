"""Audited observation-only CAD variants with stable IDs on ALL entities.

Reference programs, native geometry, split identities and action targets remain
unchanged. No TEST task is opened. Validation packages have no training export.
The changed image/instruction protocol must be reported separately.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

from cad_env.agent.observations import agent_messages

from .audit import audit_dataset
from .perception_data import sha
from .visual_tasks import render_goal

HINT = " Every hole has a black boxed integer equal to its CURRENT sketch circle index. For boxed label i use parameter diameter_i. Only colored crosshairs indicate the requested holes; black boxes label ALL holes and do not indicate selection."


def transform_task(task, canonical=False):
    task = copy.deepcopy(task)
    if task["split"] not in ("train", "validation") or task["mode"] != "edit":
        raise ValueError("Only TRAIN/validation edit observations may be transformed")
    if task["goal_spec"].get("entity_labels"):
        raise ValueError("Source already has labels")
    task["goal_spec"]["entity_labels"] = True
    if canonical:
        task["goal_spec"]["angle"] = 0
    task["instruction"] += HINT
    task["public_task"]["instruction"] = task["instruction"]
    return task


def feedback_prompt(row, task):
    blocks = row["prompt"][-1]["content"]
    if not isinstance(blocks, list) or len(row["images"]) != 1:
        raise ValueError("Expected single-goal multimodal feedback")
    context, _ = json.JSONDecoder().raw_decode(blocks[0]["text"])
    observation = {
        "state": context["state"],
        "remaining_steps": context["remaining_steps"],
        "goal_images": row["images"],
        "images": [],
    }
    return agent_messages(
        task, observation, context.get("recent_transitions", []), visual=True
    )


def build(source, output, split="validation", canonical=False):
    if split not in ("train", "validation"):
        raise ValueError("TEST transformations are forbidden")
    source, out = Path(source).resolve(), Path(output).resolve()
    audit = audit_dataset(source)
    if not audit["ok"]:
        raise ValueError(audit["errors"])
    out.mkdir(parents=True, exist_ok=False)
    manifest, hashes, tasks = [], {}, {}
    for entry in map(json.loads, (source / "manifest.jsonl").read_text().splitlines()):
        if not entry["accepted"] or entry["split"] != split:
            continue
        task_id = entry["task_id"]
        folder = source / "episodes" / task_id
        original = json.loads((folder / "task.json").read_text())
        task = transform_task(original, canonical)
        native = json.loads((folder / "oracle/result.json").read_text())
        sketch = next(
            o for o in native["initial_observation"]["objects"] if o.get("circles")
        )
        circles = sketch["circles"]
        if len(circles) != len(task["goal_spec"]["circles"]) or any(
            c["index"] != i
            or any(
                abs(c[k] - task["goal_spec"]["circles"][i][k]) > 1e-6
                for k in ("x", "y", "radius")
            )
            for i, c in enumerate(circles)
        ):
            raise ValueError("Native/image entity identity mismatch")
        dest = out / "episodes" / task_id
        (dest / "oracle").mkdir(parents=True)
        for rel in ("oracle/result.json", "oracle/output.FCStd"):
            shutil.copyfile(folder / rel, dest / rel)
        (dest / "task.json").write_text(json.dumps(task, indent=2))
        render_goal(task, dest / "goal.png")
        for rel in (
            "task.json",
            "goal.png",
            "oracle/result.json",
            "oracle/output.FCStd",
        ):
            hashes[str((dest / rel).relative_to(out))] = sha(dest / rel)
        manifest.append(entry)
        tasks[task_id] = task
    if not manifest:
        raise ValueError("Empty selected split")
    (out / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in manifest))
    hashes["manifest.jsonl"] = sha(out / "manifest.jsonl")
    counts, export_hashes = {}, {}
    if split == "train":
        rows = []
        for row in map(
            json.loads,
            (source / "exports/train.feedback.jsonl").read_text().splitlines(),
        ):
            if row["task_id"] not in tasks:
                raise ValueError("Unknown training lineage")
            row = copy.deepcopy(row)
            row["prompt"] = feedback_prompt(row, tasks[row["task_id"]])
            rows.append(row)
        (out / "exports").mkdir()
        export = out / "exports/train.feedback.jsonl"
        export.write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts = {"train.feedback": len(rows)}
        export_hashes = {"exports/train.feedback.jsonl": sha(export)}
    original = json.loads((source / "summary.json").read_text())
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "assisted-observation-v1",
        "action_contract": "single_action",
        "attempted": len(manifest),
        "accepted": len(manifest),
        "rejected": 0,
        "runtime": original["runtime"],
        "counts": counts,
        "export_sha256": export_hashes,
        "artifact_sha256": hashes,
        "source_root": str(source),
        "source_summary_sha256": sha(source / "summary.json"),
        "split": split,
        "canonical": canonical,
        "test_opened": False,
        "supported_training_exports": ["train.feedback.jsonl"]
        if split == "train"
        else [],
        "scope": "Observation-only intervention: IDs on ALL circles; original targets and native geometry; separate protocol, not weight improvement",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_assisted(out)
    (out / "observation-audit.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise ValueError(report["errors"])
    return report


def audit_assisted(root, base_report=None):
    root = Path(root).resolve()
    base = base_report or audit_dataset(root)
    errors = list(base["errors"])
    summary = json.loads((root / "summary.json").read_text())
    source = Path(summary["source_root"])
    if sha(source / "summary.json") != summary["source_summary_sha256"]:
        errors.append("Source identity mismatch")
    task_map = {}
    for entry in map(json.loads, (root / "manifest.jsonl").read_text().splitlines()):
        if entry["split"] != summary["split"] or entry["split"] not in (
            "train",
            "validation",
        ):
            errors.append("Split mismatch")
            continue
        task_id = entry["task_id"]
        original = json.loads((source / "episodes" / task_id / "task.json").read_text())
        task = json.loads((root / "episodes" / task_id / "task.json").read_text())
        if task != transform_task(original, summary["canonical"]):
            errors.append("Non-observation task mutation")
        if (
            entry["split_group"] != task["split_group"]
            or entry["split"] != task["split"]
        ):
            errors.append("Lineage mismatch")
        for rel in ("oracle/output.FCStd", "oracle/result.json"):
            if sha(root / "episodes" / task_id / rel) != sha(
                source / "episodes" / task_id / rel
            ):
                errors.append("Changed native reference")
        # Regenerate pixels, not just check caller-supplied hashes.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "goal.png"
            render_goal(task, image)
            from PIL import Image, ImageChops

            with (
                Image.open(image) as a,
                Image.open(root / "episodes" / task_id / "goal.png") as b,
            ):
                if (
                    a.size != b.size
                    or ImageChops.difference(
                        a.convert("RGB"), b.convert("RGB")
                    ).getbbox()
                ):
                    errors.append("Incorrect rendered observation")
        task_map[task_id] = task
    if summary["split"] == "train":
        source_rows = list(
            map(
                json.loads,
                (source / "exports/train.feedback.jsonl").read_text().splitlines(),
            )
        )
        rows = list(
            map(
                json.loads,
                (root / "exports/train.feedback.jsonl").read_text().splitlines(),
            )
        )
        if len(rows) != len(source_rows):
            errors.append("Missing training targets")
        for row, old in zip(rows, source_rows):
            expected = copy.deepcopy(old)
            expected["prompt"] = feedback_prompt(old, task_map[old["task_id"]])
            if row != expected:
                errors.append(
                    "Feedback differs from causal public observation transform"
                )
    elif (root / "exports").exists():
        errors.append("Validation variant must not export training rows")
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "tasks": len(task_map),
        "parents": len({t["split_group"] for t in task_map.values()}),
        "canonical": summary["canonical"],
        "test_opened": False,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--canonical", action="store_true")
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.split, a.canonical), indent=2))


if __name__ == "__main__":
    main()
