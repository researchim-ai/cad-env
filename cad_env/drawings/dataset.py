"""Observation-only conversion of verified CAD corpora to technical drawing inputs.

Private CAD teachers stay unchanged. Public prompts are reconstructed from their
causal state/history and receive only verified views and readable document crops.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

from cad_env.agent.observations import agent_messages
from cad_env.learning.audit import audit_dataset
from cad_env.learning.factory import jsonl

from .export import audit_drawing, export_drawing, sha
from .profiles import get_profile, validate_metadata
from .sheets import PROFILE

GUIDE = (
    "The goal images are first-angle drawing views and labeled schedule crops. "
    "Match feature IDs between position and size/depth crops. All dimensions are in millimeters. "
)


def transformed_task(source, bundle, images, standard="nominal"):
    task = copy.deepcopy(source)
    task.pop("goal_spec", None)
    task["instruction"] = GUIDE + source["instruction"]
    task["public_task"]["instruction"] = task["instruction"]
    task["goal_images"] = images
    profile = get_profile(standard)
    task["drawing_profile"] = profile.profile_id if profile else PROFILE
    if profile:
        task["drawing_standard"] = standard
    task["drawing_bundle"] = bundle
    return task


def transformed_row(source, task):
    row = copy.deepcopy(source)
    content = row["prompt"][-1]["content"]
    original = content if isinstance(content, str) else content[0]["text"]
    context, _ = json.JSONDecoder().raw_decode(original)
    observation = {
        "state": context["state"],
        "remaining_steps": context["remaining_steps"],
        "goal_images": task["goal_images"],
        "images": [],
    }
    if "last_error" in context:
        observation["error"] = context["last_error"]
    row["prompt"] = agent_messages(
        task, observation, context.get("recent_transitions", []), visual=True
    )
    row["images"] = task["goal_images"]
    row["goal_required"] = True
    return row


def convert_dataset(
    source, output, *, standard="nominal", metadata=None, require_conformance=False
):
    source, out = Path(source).resolve(), Path(output).resolve()
    profile = get_profile(standard)
    metadata = validate_metadata(metadata or {})
    source_summary = json.loads((source / "summary.json").read_text())
    entries = [
        json.loads(s)
        for s in (source / "manifest.jsonl").read_text().splitlines()
        if s.strip()
    ]
    if any(e["split"] == "test" for e in entries):
        raise ValueError(
            "Use a TRAIN/validation-only source package; TEST geometry is not opened"
        )
    audit = audit_dataset(source)
    if not audit["ok"]:
        raise ValueError("Source corpus audit failed")
    entries = [
        e for e in entries if e["accepted"] and e["split"] in ("train", "validation")
    ]
    out.mkdir(parents=True, exist_ok=False)
    bundles = {}
    tasks = {}
    manifest = []
    for index, entry in enumerate(entries):
        folder = source / "episodes" / entry["task_id"]
        original = json.loads((folder / "task.json").read_text())
        if not original.get("goal_only_observation"):
            raise ValueError("Current-state render conversion needs its own adapter")
        group = original["split_group"]
        signature = json.dumps(
            [original["parameters"], original["reference_actions"]], sort_keys=True
        )
        if group not in bundles:
            relative = "drawings/" + group
            print(
                f"Native drawing {len(bundles) + 1}: {original['family']} {original.get('difficulty', '')}",
                flush=True,
            )
            result = export_drawing(
                folder / "oracle/output.FCStd",
                out / relative,
                task=original,
                standard=standard,
                metadata=metadata,
                require_conformance=require_conformance,
            )
            bundles[group] = (signature, relative, result)
        signature_before, relative, result = bundles[group]
        if signature_before != signature:
            raise ValueError("One drawing lineage contains different target geometry")
        images = [relative + "/" + p for p in result["vlm_tiles"]]
        task = transformed_task(original, relative, images, standard)
        target = out / "episodes" / task["task_id"]
        (target / "oracle").mkdir(parents=True)
        for name in ("output.FCStd", "result.json"):
            shutil.copyfile(folder / "oracle" / name, target / "oracle" / name)
        shutil.copyfile(folder / "trajectory.jsonl", target / "trajectory.jsonl")
        shutil.copyfile(folder / "task.json", target / "source-task.json")
        (target / "task.json").write_text(json.dumps(task, indent=2))
        manifest.append(entry)
        tasks[task["task_id"]] = task
    counts = {}
    for split in ("train", "validation"):
        original_rows = [
            json.loads(s)
            for s in (source / "exports" / f"{split}.feedback.jsonl")
            .read_text()
            .splitlines()
            if s.strip()
        ]
        original_rows = [r for r in original_rows if r["task_id"] in tasks]
        jsonl(out / "source-exports" / f"{split}.feedback.jsonl", original_rows)
        converted = [
            transformed_row(row, tasks[row["task_id"]]) for row in original_rows
        ]
        jsonl(out / "exports" / f"{split}.feedback.jsonl", converted)
        counts[split + ".feedback"] = len(converted)
    jsonl(out / "manifest.jsonl", manifest)
    summary = {
        **source_summary,
        "benchmark": "technical-drawing-v1",
        "drawing_profile": profile.profile_id if profile else PROFILE,
        "drawing_standard": standard,
        "standards_claim": "No complete conformance claim; inspect each drawing conformance.json",
        "source_summary_sha256": sha(source / "summary.json"),
        "source_dataset": str(source),
        "attempted": len(manifest),
        "accepted": len(manifest),
        "rejected": 0,
        "counts": counts,
        "standards_conformance": (
            "nonconforming"
            if any(
                v[2]["standards_conformance"] == "nonconforming"
                for v in bundles.values()
            )
            else "partially_verified"
            if profile
            else "not_assessed"
        ),
        "test_opened": False,
        "drawing_bundles": len(bundles),
        "supported_training_exports": ["train.feedback.jsonl"],
        "export_sha256": {
            str(p.relative_to(out)): sha(p) for p in (out / "exports").glob("*.jsonl")
        },
        "artifact_sha256": {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and ".runtime" not in p.parts
        },
        "scope": "Same native tasks and teachers with verified model-derived drawings. Observation-format conversion is not a model-weight gain.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = audit_technical_dataset(out)
    (out / "drawing-audit.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise ValueError(
            "Converted drawing corpus audit failed: " + str(report["errors"][:3])
        )
    return summary


def audit_technical_dataset(root):
    root = Path(root).resolve()
    metadata = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if any(e["split"] == "test" for e in metadata):
        return {
            "ok": False,
            "errors": ["TEST geometry is not opened during drawing development audit"],
        }
    report = audit_dataset(root)
    errors = list(report["errors"])
    entries = [
        json.loads(s)
        for s in (root / "manifest.jsonl").read_text().splitlines()
        if s.strip()
    ]
    tasks = {}
    bundles = set()
    for entry in entries:
        folder = root / "episodes" / entry["task_id"]
        task = json.loads((folder / "task.json").read_text())
        source = json.loads((folder / "source-task.json").read_text())
        bundle = (root / task["drawing_bundle"]).resolve()
        if not bundle.is_relative_to(root):
            raise ValueError("Drawing bundle escapes corpus")
        data = json.loads((bundle / "manifest.json").read_text())
        images = [task["drawing_bundle"] + "/" + p for p in data["vlm_tiles"]]
        if task != transformed_task(
            source,
            task["drawing_bundle"],
            images,
            task.get("drawing_standard", "nominal"),
        ):
            errors.append(
                "Task/teacher changed beyond observation format: " + entry["task_id"]
            )
        if entry["split"] == "test":
            errors.append("TEST exported during drawing development")
        if bundle not in bundles:
            bundles.add(bundle)
            drawing = audit_drawing(bundle)
            errors.extend(drawing["errors"])
        tasks[task["task_id"]] = task
    for split in ("train", "validation"):
        expected = [
            transformed_row(json.loads(line), tasks[json.loads(line)["task_id"]])
            for line in (root / "source-exports" / f"{split}.feedback.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        actual = [
            json.loads(line)
            for line in (root / "exports" / f"{split}.feedback.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        if actual != expected:
            errors.append(
                "Causal prompt/label/image conversion differs from source: " + split
            )
    return {
        **report,
        "ok": not errors,
        "errors": errors,
        "drawing_bundles_checked": len(bundles),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--standard", choices=["nominal", "iso", "gost"], default="nominal")
    p.add_argument("--metadata")
    p.add_argument("--require-conformance", action="store_true")
    a = p.parse_args()
    result = convert_dataset(
        a.source,
        a.output,
        standard=a.standard,
        metadata=json.loads(Path(a.metadata).read_text()) if a.metadata else None,
        require_conformance=a.require_conformance,
    )
    print(json.dumps({k: result[k] for k in ("accepted", "drawing_bundles", "counts")}))


if __name__ == "__main__":
    main()
