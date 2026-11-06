"""Audited TRAIN-only complete-plan supervision from native-verified tasks."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from cad_env.agent.observations import agent_messages

from .perception_plan import PublicCropper, compile_plan, plan_messages


def teacher_plan(task):
    if task["mode"] == "edit":
        edits = [a for a in task["oracle_actions"] if a["op"] == "set_parameter"]
        if (
            not edits
            or len({a["value"] for a in edits}) != 1
            or len({a["target"] for a in edits}) != 1
        ):
            raise ValueError("Complete selection plan requires common target/value")
        return {
            "kind": "selection",
            "target": edits[0]["target"],
            "parameters": sorted(
                [a["parameter"] for a in edits], key=lambda x: int(x.split("_")[-1])
            ),
            "value": edits[0]["value"],
        }
    p = task["parameters"]
    if task["family"] == "counterbored_plate":
        blind = p["holes"][-1]
        return {
            "kind": "counterbored_plate",
            "base": [p["length"], p["width"], p["height"]],
            "centers": [[h["x"], h["y"]] for h in p["holes"][:3]],
            "through_diameter": 2 * p["bore_radius"],
            "counterbore_diameter": 2 * p["counter_radius"],
            "counterbore_depth": p["counter_depth"],
            "blind": [blind["x"], blind["y"], 2 * blind["radius"], blind["depth"]],
        }
    if task["family"] == "hollow_stacked_shaft":
        return {
            "kind": "hollow_stacked_shaft",
            "stages": [[2 * s["radius"], s["height"]] for s in p["stages"]],
            "bore_diameter": 2 * p["bore_radius"],
            "mount_offset": p["stages"][0]["radius"] - 2.2,
            "mount_diameter": 2 * p["mount_radius"],
        }
    raise ValueError("Unsupported plan teacher family")


def public_initial(root, task):
    native = json.loads(
        (root / "episodes" / task["task_id"] / "oracle/result.json").read_text()
    )
    return agent_messages(
        task,
        {
            "state": native["initial_observation"],
            "remaining_steps": task["max_steps"],
            "goal_images": [str(root / "episodes" / task["task_id"] / "goal.png")],
            "images": [],
        },
        visual=True,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output):
    from .train import check_training_source

    source, out = Path(source).resolve(), Path(output).resolve()
    check_training_source(source, source / "exports/train.feedback.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    cropper = PublicCropper(out / "public-crops")
    manifest = []
    rows = []
    hashes = {}
    for line in (source / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"] or entry["split"] != "train":
            continue
        task = json.loads(
            (source / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        if task.get("goal_spec", {}).get("entity_labels"):
            continue
        if task["mode"] != "edit" and task["family"] not in {
            "counterbored_plate",
            "hollow_stacked_shaft",
        }:
            continue
        plan = teacher_plan(task)
        for relative in [
            "task.json",
            "goal.png",
            "oracle/result.json",
            "oracle/output.FCStd",
        ]:
            src = source / "episodes" / entry["task_id"] / relative
            dst = out / "episodes" / entry["task_id"] / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            hashes[str(dst.relative_to(out))] = sha(dst)
        public = public_initial(out, task)
        images = cropper.images(
            public, [out / "episodes" / entry["task_id"] / "goal.png"]
        )
        prompt = plan_messages(public, len(images))
        actions = compile_plan(plan, public)
        if task["mode"] == "create" and actions != task["oracle_actions"]:
            raise ValueError("Plan compiler does not reproduce verified teacher")
        rows.append(
            {
                "task_id": entry["task_id"],
                "split_group": entry["split_group"],
                "step_id": 0,
                "prompt": prompt,
                "completion": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(plan)}],
                    }
                ],
                "images": [str(Path(p).relative_to(out)) for p in images],
            }
        )
        manifest.append(entry)
    if not rows:
        raise ValueError("No eligible TRAIN plan rows")

    def write_jsonl(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in data))

    write_jsonl(out / "manifest.jsonl", manifest)
    write_jsonl(out / "exports/train.feedback.jsonl", rows)
    for path in (out / "public-crops").glob("*.png"):
        hashes[str(path.relative_to(out))] = sha(path)
    hashes["manifest.jsonl"] = sha(out / "manifest.jsonl")
    original = json.loads((source / "summary.json").read_text())
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "perception-plan-v1",
        "action_contract": "perception_plan",
        "attempted": len(rows),
        "accepted": len(rows),
        "rejected": 0,
        "counts": {"train.feedback": len(rows)},
        "artifact_sha256": hashes,
        "export_sha256": {
            "exports/train.feedback.jsonl": sha(out / "exports/train.feedback.jsonl")
        },
        "runtime": original["runtime"],
        "supported_training_exports": ["train.feedback.jsonl"],
        "source_summary_sha256": sha(source / "summary.json"),
        "source_root": str(source),
        "scope": "TRAIN-only complete perception plans; public pixel crops; deterministic tool executor is a different policy contract",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_plans(out)
    (out / "plan-audit.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise ValueError(str(report["errors"]))
    return report


def audit_plans(root, base_report=None):
    from .audit import audit_dataset

    root = Path(root)
    base = base_report or audit_dataset(root)
    errors = list(base["errors"])
    for line in (root / "exports/train.feedback.jsonl").read_text().splitlines():
        row = json.loads(line)
        task = json.loads(
            (root / "episodes" / row["task_id"] / "task.json").read_text()
        )
        if task["split"] != "train" or row["split_group"] != task["split_group"]:
            errors.append("Held-out/lineage mismatch")
        public = public_initial(root, task)
        if row["prompt"] != plan_messages(public, len(row["images"])):
            errors.append("Prompt differs from public-only contract")
        completion = json.loads(row["completion"][0]["content"][0]["text"])
        if completion != teacher_plan(task):
            errors.append("Incorrect complete-plan target")
        actions = compile_plan(completion, public)
        if task["mode"] == "create" and actions != task["oracle_actions"]:
            errors.append("Compiler/teacher mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "contract": "perception_plan",
        "train_only": True,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output)))


if __name__ == "__main__":
    main()
