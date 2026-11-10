"""Publish only audited student FCStd examples, never teacher replacements."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.rendering import render_mesh


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def showcase_groups(rows):
    """Select by level/family/ID, never replace a student's failed output."""
    result = []
    for level in ("easy", "medium", "hard"):
        families = (
            ("terraced_flange_stack",)
            if level == "hard"
            else ("heterogeneous_plate", "multi_support_fixture")
        )
        for family in families:
            groups = {}
            for row in rows:
                if (
                    row["task_success"]
                    and row["difficulty"] == level
                    and row["family"] == family
                ):
                    groups.setdefault(row["task_id"], []).append(row)
            count = 2 if level == "hard" else 1
            if len(groups) < count:
                raise ValueError("Missing requested showcase coverage")
            result.extend((key, groups[key]) for key in sorted(groups)[:count])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reuse-previews", action="store_true")
    args = parser.parse_args()
    benchmark = ROOT / "artifacts/curriculum-benchmark-90-v1"
    run = benchmark / "checkpoint250"
    status = json.loads((run / "status.json").read_text())
    audit = json.loads(
        (ROOT / "artifacts/curriculum-solutions-audit-v1/status.json").read_text()
    )
    if status["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Completed audited run required")
    all_rows = status["models"]["tuned-iso-gost"]["episodes"]
    selected_groups = showcase_groups(all_rows)
    rows = [r for _, cases in selected_groups for r in cases]
    selected = {
        p["task_id"]: p for p in json.loads((run / "selection.json").read_text())
    }
    out = ROOT / "web/assets/student-solutions"
    out.mkdir(parents=True, exist_ok=True)
    previous = (
        json.loads((out / "manifest.json").read_text()) if args.reuse_previews else None
    )
    baseline_rows = json.loads((benchmark / "baseline/status.json").read_text())[
        "models"
    ]["qwen35-base"]["episodes"]
    baseline = {(r["task_id"], r["standard"]): r for r in baseline_rows}
    examples = []
    native_jobs = []
    reuse = set()
    level_counts = {}
    for task_id, cases in selected_groups:
        level = cases[0]["difficulty"]
        level_counts[level] = level_counts.get(level, 0) + 1
        slug = f"{'flange' if level == 'hard' else level}-{level_counts[level]:02d}"
        task = json.loads(
            Path(selected[task_id]["representations"]["iso"]["task"]).read_text()
        )
        example = {
            "id": slug,
            "task_id": task_id,
            "family": task["family"],
            "split": "validation",
            "pads": len(task["parameters"]["pads"]),
            "holes": len(task["parameters"]["cuts"]),
            "steps": next(
                (r["steps"] for r in cases if r["standard"] == "iso"), cases[0]["steps"]
            ),
            "difficulty": level,
            "bounds_mm": task["analytic_metrics"]["bounds"],
            "solutions": {},
            "baseline": {},
        }
        for standard in ("iso", "gost"):
            base = baseline[task_id, standard]
            basecase = Path(base["episode_dir"]).parent.parent
            grade = basecase / "final-score"
            if base["stop_reason"] == "agent_finish":
                ev = json.loads(
                    (Path(base["episode_dir"]) / "trajectory.jsonl")
                    .read_text()
                    .splitlines()[-1]
                )
                grade = Path(ev["info"]["result_dir"])
            source = grade / "output.FCStd"
            native = json.loads((grade / "result.json").read_text())
            if sha(source) != native["artifact_sha256"]:
                raise ValueError("Baseline artifact changed")
            filename = f"{slug}-base-{standard}.FCStd"
            actions = f"{slug}-base-{standard}-actions.json"
            shutil.copyfile(source, out / filename)
            (out / actions).write_text(json.dumps(base["committed_actions"], indent=2))
            example["baseline"][standard] = {
                "file": filename,
                "sha256": sha(out / filename),
                "actions": actions,
                "actions_sha256": sha(out / actions),
                "steps": base["steps"],
                "committed_actions": len(base["committed_actions"]),
                "task_success": base["task_success"],
                "shape_valid": base["validation"]["shape_valid"],
                "source": str(source.relative_to(ROOT)),
            }
        for case in sorted(cases, key=lambda r: r["standard"]):
            events = [
                json.loads(s)
                for s in (Path(case["episode_dir"]) / "trajectory.jsonl")
                .read_text()
                .splitlines()
            ]
            grade = Path(events[-1]["info"]["result_dir"])
            source = grade / "output.FCStd"
            native = json.loads((grade / "result.json").read_text())
            if (
                sha(source) != native["artifact_sha256"]
                or case["validation"]["relative_symmetric_difference"] != 0
            ):
                raise ValueError("Student artifact provenance/geometry mismatch")
            filename = f"{slug}-{case['standard']}.FCStd"
            shutil.copyfile(source, out / filename)
            actions = f"{slug}-{case['standard']}-actions.json"
            (out / actions).write_text(json.dumps(case["committed_actions"], indent=2))
            example["solutions"][case["standard"]] = {
                "file": filename,
                "sha256": sha(out / filename),
                "actions": actions,
                "actions_sha256": sha(out / actions),
                "steps": case["steps"],
                "source": str(source.relative_to(ROOT)),
                "geometry_error": 0.0,
            }
        example["inputs"] = {}
        for standard in ("iso", "gost"):
            representation = selected[task_id]["representations"][standard]
            model_case = next(
                r
                for r in all_rows
                if r["task_id"] == task_id and r["standard"] == standard
            )
            response_file = (
                Path(model_case["episode_dir"]).parent.parent / "responses.jsonl"
            )
            response = json.loads(response_file.read_text().splitlines()[0])
            images = representation["vlm_images"]
            if [sha(p) for p in images] != response["image_sha256"]:
                raise ValueError("Input drawings differ from actual model request")
            records = []
            metadata = json.loads(
                (Path(images[0]).parent.parent / "tiles.json").read_text()
            )["tiles"]
            for index, source in enumerate(images, 1):
                name = f"{slug}-input-{standard}-{index:02d}.png"
                shutil.copyfile(source, out / name)
                records.append(
                    {
                        "file": name,
                        "sha256": sha(out / name),
                        "ordinal": index,
                        "caption": metadata[index - 1]["caption"],
                        "source": str(Path(source).relative_to(ROOT)),
                    }
                )
            example["inputs"][standard] = {
                "images": records,
                "task_success": model_case["task_success"],
                "input_role": "actual ordered goal images from the model request; no current-candidate images",
            }
        chosen = example["solutions"].get("iso") or example["solutions"]["gost"]
        example["rendered_from"] = chosen["file"]
        if previous:
            old = next(
                (
                    e
                    for e in previous["examples"]
                    if e["task_id"] == task_id and e["id"] == slug
                ),
                None,
            )
            old_rep = (
                next(
                    (
                        r
                        for r in old["solutions"].values()
                        if r["file"] == chosen["file"]
                    ),
                    None,
                )
                if old
                else None
            )
            if old_rep and old_rep["sha256"] != chosen["sha256"]:
                raise ValueError("Cannot reuse previews of changed student output")
            if old_rep and all(
                sha(out / p["file"]) == p["sha256"] for p in old["previews"].values()
            ):
                reuse.add(slug)
        if slug not in reuse:
            native_jobs.append(
                {
                    "source": str(out / chosen["file"]),
                    "output": str(out / (slug + "-mesh.json")),
                }
            )
        examples.append(example)
    with tempfile.TemporaryDirectory(prefix="cad-student-previews-") as tmp:
        folder = Path(tmp)
        request = folder / "request.json"
        request.write_text(json.dumps(native_jobs))
        script = folder / "export.py"
        script.write_text(
            """import FreeCAD as App,json
from pathlib import Path
for job in json.loads(Path("""
            + repr(str(request))
            + """).read_text()):
 doc=App.openDocument(job['source']);doc.recompute()
 bodies=[o for o in doc.Objects if o.TypeId=='PartDesign::Body']
 if len(bodies)!=1 or not bodies[0].Shape.isValid():raise ValueError('Invalid student part')
 shape=bodies[0].Shape
 vertices,faces=shape.tessellate(0.3)
 points=[[v.x,v.y,v.z] for v in vertices]
 # Unreferenced bounding vertices reserve camera padding; the rendered native
 # triangles remain unchanged. Tall support fixtures otherwise touch the edge.
 lower=[min(v[i] for v in points) for i in range(3)]
 upper=[max(v[i] for v in points) for i in range(3)]
 center=[(a+b)/2 for a,b in zip(lower,upper)]
 points.extend([[c+1.2*(v-c) for c,v in zip(center,bound)] for bound in (lower,upper)])
 Path(job['output']).write_text(json.dumps({'vertices':points,'faces':[list(f) for f in faces]}))
 App.closeDocument(doc.Name)
"""
        )
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
        for k in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
            env[k] = str(folder / k)
        if native_jobs:
            subprocess.run(
                [FreeCADBackend().executable, str(script)],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                timeout=180,
            )
    for example in examples:
        slug = example["id"]
        paths = (
            [str(out / f"{slug}-{view}.png") for view in ("iso", "top")]
            if slug in reuse
            else render_mesh(
                out / (slug + "-mesh.json"),
                out,
                prefix=slug,
                size=720,
                views=("iso", "top"),
            )
        )
        example["previews"] = {
            view: {"file": Path(p).name, "sha256": sha(p)}
            for view, p in zip(("iso", "top"), paths)
        }
        # Mesh is only an intermediate renderer input, not needed by the page.
        (out / (slug + "-mesh.json")).unlink(missing_ok=True)
    report = {
        "scope": "Two illustrative successful designs per difficulty; easy and medium include plate/fixture families, hard shows two flange variants. Selection is not the aggregate success rate.",
        "preview_framing_margin": 1.2,
        "examples": examples,
        "result_unit": "90 parent tasks, ISO and GOST representations reported separately",
        "results_by_representation": {
            standard: {
                "tasks": 90,
                "successes": sum(
                    r["task_success"] for r in all_rows if r["standard"] == standard
                ),
            }
            for standard in ("iso", "gost")
        },
        "paired_task_outcomes": {
            "both": sum(
                all(r["task_success"] for r in all_rows if r["task_id"] == task)
                for task in selected
            ),
            "iso_only": sum(
                next(
                    r["task_success"]
                    for r in all_rows
                    if r["task_id"] == task and r["standard"] == "iso"
                )
                and not next(
                    r["task_success"]
                    for r in all_rows
                    if r["task_id"] == task and r["standard"] == "gost"
                )
                for task in selected
            ),
            "gost_only": sum(
                next(
                    r["task_success"]
                    for r in all_rows
                    if r["task_id"] == task and r["standard"] == "gost"
                )
                and not next(
                    r["task_success"]
                    for r in all_rows
                    if r["task_id"] == task and r["standard"] == "iso"
                )
                for task in selected
            ),
            "neither": sum(
                not any(r["task_success"] for r in all_rows if r["task_id"] == task)
                for task in selected
            ),
        },
        "audit_source_sha256": sha(
            ROOT / "artifacts/curriculum-solutions-audit-v1/status.json"
        ),
        "comparison": json.loads(
            (benchmark / "comparison-checkpoint250.json").read_text()
        ),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    keep = {"manifest.json"}
    for example in examples:
        for group in ("solutions", "baseline"):
            for rep in example[group].values():
                keep.update((rep["file"], rep["actions"]))
        keep.update(p["file"] for p in example["previews"].values())
        keep.update(
            im["file"] for rep in example["inputs"].values() for im in rep["images"]
        )
    # Remove superseded generated gallery files only. Originals remain in the
    # benchmark; the site only contains its current individual examples.
    for path in out.iterdir():
        if (
            path.is_file()
            and path.suffix in (".FCStd", ".json", ".png", ".zip")
            and path.name not in keep
        ):
            path.unlink()
    print(
        "Exported",
        len(examples),
        "geometries",
        len(rows),
        "student FCStd files",
        flush=True,
    )


if __name__ == "__main__":
    main()
