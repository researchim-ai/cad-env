"""TRAIN-only auxiliary visual supervision; never a CAD action contract.

Prompts contain pixels and public initial geometry. Private task labels are used
only to create completion targets and to audit the export, never as model inputs.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from .perception_data import public_initial, sha
from .perception_plan import public_payload
from .perception_probe import messages

COUNT = 'Count every COLORED crosshair marker visible in the image. Count markers, not black hole outlines. Return {"count": integer}.'
SELECT = 'Match ALL colored crosshairs in the image to the CURRENT sketch circle indices below. Use the drawn X/Y arrows for rotation. Include no unmarked circle and omit no marked circle. Return {"indices": [integers sorted ascending]}. Current public circles: '
TABLE = 'Read EVERY row of the visible HOLES table as [X,Y,DIAMETER,DEPTH]. Preserve separate rows that share XY but have a different diameter or depth. Include all rows, without generating CAD code. Return {"holes": [[x,y,diameter,depth],...]}. All values are mm.'


def targets(task, public):
    """Build auxiliary questions, never inject their answers into the question."""
    if task["mode"] == "edit":
        if task.get("goal_spec", {}).get("entity_labels"):
            return []
        selected = sorted(task["goal_spec"]["selected"])
        sketch = next(
            o for o in public_payload(public)["state"]["objects"] if o.get("circles")
        )
        circles = [
            {k: c[k] for k in ("index", "x", "y", "radius")} for c in sketch["circles"]
        ]
        indices = [c["index"] for c in circles]
        if len(set(selected)) != len(selected) or not set(selected) <= set(indices):
            raise ValueError("Invalid selected circle set")
        return [
            ("count", COUNT, {"count": len(selected)}),
            ("selection", SELECT + json.dumps(circles), {"indices": selected}),
        ]
    if task["family"] == "counterbored_plate" and task.get("input_modality") == "image":
        return [
            (
                "hole_table",
                TABLE,
                {
                    "holes": [
                        [h["x"], h["y"], 2 * h["radius"], h["depth"]]
                        for h in task["parameters"]["holes"]
                    ]
                },
            )
        ]
    return []


def build(source, output):
    from .train import check_training_source

    source, out = Path(source).resolve(), Path(output).resolve()
    check_training_source(source, source / "exports/train.feedback.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    rows, manifest, hashes = [], [], {}
    seen_images = set()
    for line in (source / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"] or entry["split"] != "train":
            continue
        folder = source / "episodes" / entry["task_id"]
        task = json.loads((folder / "task.json").read_text())
        public = public_initial(source, task)
        questions = targets(task, public)
        # Identical mm/inch pixels teach the same VQA target: do not upweight them.
        image_hash = sha(folder / "goal.png")
        if not questions or image_hash in seen_images:
            continue
        seen_images.add(image_hash)
        for rel in (
            "task.json",
            "goal.png",
            "oracle/result.json",
            "oracle/output.FCStd",
        ):
            dst = out / "episodes" / entry["task_id"] / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / rel, dst)
            hashes[str(dst.relative_to(out))] = sha(dst)
        image = f"episodes/{entry['task_id']}/goal.png"
        for kind, question, answer in questions:
            rows.append(
                {
                    "task_id": entry["task_id"],
                    "split_group": entry["split_group"],
                    "step_id": len(rows),
                    "auxiliary_kind": kind,
                    "prompt": messages(question, [image]),
                    "completion": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": json.dumps(answer)}],
                        }
                    ],
                    "images": [image],
                }
            )
        manifest.append(entry)
    if not rows:
        raise ValueError("No eligible TRAIN examples")
    (out / "exports").mkdir()
    for path, records in [
        (out / "manifest.jsonl", manifest),
        (out / "exports/train.feedback.jsonl", rows),
    ]:
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    hashes["manifest.jsonl"] = sha(out / "manifest.jsonl")
    original = json.loads((source / "summary.json").read_text())
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "auxiliary-grounding-v1",
        "action_contract": "auxiliary_grounding",
        "attempted": len(manifest),
        "accepted": len(manifest),
        "rejected": 0,
        "counts": {"train.feedback": len(rows)},
        "runtime": original["runtime"],
        "artifact_sha256": hashes,
        "export_sha256": {
            "exports/train.feedback.jsonl": sha(out / "exports/train.feedback.jsonl")
        },
        "supported_training_exports": ["train.feedback.jsonl"],
        "source_summary_sha256": sha(source / "summary.json"),
        "source_root": str(source),
        "kinds": dict(Counter(r["auxiliary_kind"] for r in rows)),
        "independent_parents": len({r["split_group"] for r in rows}),
        "distinct_images": len(seen_images),
        "scope": "TRAIN auxiliary VQA; no deterministic CAD compiler; preserve single-action evaluation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_grounding(out)
    (out / "grounding-audit.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise ValueError(report["errors"])
    return {
        **report,
        "kinds": summary["kinds"],
        "independent_parents": summary["independent_parents"],
        "distinct_images": len(seen_images),
    }


def audit_grounding(root, base_report=None):
    from .audit import audit_dataset

    root = Path(root).resolve()
    base = base_report or audit_dataset(root)
    errors = list(base["errors"])
    manifest = {
        e["task_id"]: e
        for e in map(json.loads, (root / "manifest.jsonl").read_text().splitlines())
    }
    seen = set()
    for row in map(
        json.loads, (root / "exports/train.feedback.jsonl").read_text().splitlines()
    ):
        entry = manifest.get(row["task_id"])
        if not entry or not entry["accepted"] or entry["split"] != "train":
            errors.append("Non-TRAIN or unknown task")
            continue
        folder = root / "episodes" / row["task_id"]
        task = json.loads((folder / "task.json").read_text())
        if (
            task["split"] != "train"
            or task["split_group"] != row["split_group"]
            or entry["split_group"] != row["split_group"]
        ):
            errors.append("Lineage mismatch")
        expected = {k: (q, a) for k, q, a in targets(task, public_initial(root, task))}
        kind = row["auxiliary_kind"]
        key = (row["task_id"], kind)
        if key in seen or kind not in expected:
            errors.append("Duplicate/unknown auxiliary target")
            continue
        seen.add(key)
        question, answer = expected[kind]
        images = [f"episodes/{row['task_id']}/goal.png"]
        if row["images"] != images or row["prompt"] != messages(question, images):
            errors.append("Prompt/image differs from public-only contract")
        completion = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(answer)}],
            }
        ]
        if row["completion"] != completion:
            errors.append("Incorrect grounding label")
    expected_keys = set()
    for task_id in manifest:
        task = json.loads((root / "episodes" / task_id / "task.json").read_text())
        expected_keys.update(
            (task_id, k) for k, _, _ in targets(task, public_initial(root, task))
        )
    if seen != expected_keys:
        errors.append("Missing auxiliary targets")
    return {
        "ok": not errors,
        "errors": errors,
        "base_audit": base,
        "train_only": True,
        "contract": "auxiliary_grounding",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
