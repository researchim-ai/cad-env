"""Native-verified multi-engine corpora with image-conditioned next-action SFT."""

import hashlib
import json
import shutil
from pathlib import Path

from cad_env.agent.observations import normalize_single_action_prompt
from cad_env.agent.rendering import render_mesh
from cad_env.agent.schema import ACTION_GUIDE

from .backend import MultiCADBackend
from .plan import resolve
from .tasks import FAMILIES, make_task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


def specification_image(lines, path):
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.new("RGB", (768, max(512, 90 + 30 * len(lines))), "white")
    draw = ImageDraw.Draw(picture)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (20, 12),
        "ENGINEERING SPECIFICATION / all dimensions in mm",
        fill="black",
        font=font,
    )
    # Wrap long specifications so no dimension is silently clipped.
    y = 50
    import textwrap

    wrapped = [part for line in lines for part in textwrap.wrap(line, width=65)]
    if 28 * len(wrapped) + 65 > picture.height:
        picture = Image.new("RGB", (768, 28 * len(wrapped) + 65), "white")
        draw = ImageDraw.Draw(picture)
        draw.text((20, 12), "ENGINEERING SPECIFICATION / mm", fill="black", font=font)
    for line in wrapped:
        draw.text((20, y), line, fill="black", font=font)
        y += 28
    picture.save(path)


def messages(engine, task, state, history, images):
    public = {"instruction": task["instruction"], "units": "mm", "mode": task["mode"]}
    if task["input_modality"] == "text":
        public["specification"] = task["specification"]
    else:
        public["instruction"] += (
            " Read the dimensions from the goal specification image; goal views follow it."
        )
    system = f"You control {engine} through typed CAD tools.\n" + ACTION_GUIDE
    system += "\nReturn exactly ONE JSON action object. FIRST images show the GOAL; the LAST image shows the CURRENT state. Preserve existing features when editing."
    context = {"task": public, "state": state, "recent_transitions": history[-4:]}
    return normalize_single_action_prompt(
        [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(context)}]
                + [{"type": "image"} for _ in images],
            },
        ]
    )


def generate(
    output,
    *,
    engines=("freecad", "cadquery"),
    seeds=3,
    start_seed=0,
    families=FAMILIES,
    modes=("create", "edit", "repair"),
    modalities=("text", "image"),
    python=None,
    view_mode="multiview",
    varied_edits=False,
):
    if view_mode not in {"singleview", "multiview"}:
        raise ValueError("Choose singleview or multiview")
    if seeds < 1 or start_seed < 0 or not engines or len(set(engines)) != len(engines):
        raise ValueError("Invalid generation budget/engines")
    if (
        not families
        or set(families) - set(FAMILIES)
        or not modes
        or set(modes) - {"create", "edit", "repair"}
        or not modalities
        or set(modalities) - {"text", "image"}
    ):
        raise ValueError("Invalid task configuration")
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    backends = {engine: MultiCADBackend(engine, python=python) for engine in engines}
    manifest = []
    exports = {s: [] for s in ("train", "validation", "test")}
    rejected = []
    for seed in range(start_seed, start_seed + seeds):
        for family in families:
            for mode in modes:
                for modality in modalities:
                    task = make_task(
                        family, seed, mode, modality, varied_edits=varied_edits
                    )
                    for engine, backend in backends.items():
                        identifier = (
                            "m-"
                            + hashlib.sha256(
                                json.dumps(
                                    [task["split_group"], mode, modality, engine]
                                ).encode()
                            ).hexdigest()[:24]
                        )
                        if any(row["task_id"] == identifier for row in manifest):
                            continue
                        case = out / "episodes" / identifier
                        case.mkdir(parents=True)
                        (case / "task.json").write_text(
                            json.dumps(
                                {**task, "task_id": identifier, "engine": engine},
                                indent=2,
                            )
                        )
                        try:
                            reference = backend.run(
                                task["reference_actions"],
                                case / "reference",
                                reference_actions=task["reference_actions"],
                                expected_metrics=task["analytic_metrics"],
                            )
                            result = backend.run(
                                task["oracle_actions"],
                                case / "teacher",
                                initial_actions=task["initial_actions"],
                                reference_actions=task["reference_actions"],
                                expected_metrics=task["analytic_metrics"],
                                record=True,
                            )
                            if (
                                not reference.get("execution_ok")
                                or not reference.get("validation", {}).get(
                                    "task_success"
                                )
                                or not result.get("execution_ok")
                                or not result.get("validation", {}).get("task_success")
                            ):
                                raise RuntimeError(
                                    "Native oracle failed: "
                                    + str(
                                        result.get("error") or result.get("validation")
                                    )
                                )
                            goal_dir = case / "goal"
                            goal_dir.mkdir()
                            specification_image(
                                task["specification"], goal_dir / "spec.png"
                            )
                            views = render_mesh(
                                case / "reference/mesh.json",
                                goal_dir,
                                size=384,
                                views=("iso", "top", "front", "right"),
                            )
                            # Both modalities have the same views; only text condition duplicates dimensions in text.
                            if view_mode == "multiview":
                                from PIL import Image

                                collage = Image.new("RGB", (768, 768), "white")
                                for index, view in enumerate(views):
                                    with Image.open(view) as picture:
                                        collage.paste(
                                            picture,
                                            ((index % 2) * 384, (index // 2) * 384),
                                        )
                                collage.save(goal_dir / "multiview.png")
                            goals = [
                                goal_dir / "spec.png",
                                goal_dir / "multiview.png"
                                if view_mode == "multiview"
                                else Path(views[0]),
                            ]
                            (case / "task.json").write_text(
                                json.dumps(
                                    {
                                        **task,
                                        "task_id": identifier,
                                        "engine": engine,
                                        "view_mode": view_mode,
                                        "goal_images": [
                                            str(p.relative_to(out)) for p in goals
                                        ],
                                    },
                                    indent=2,
                                )
                            )
                            events = result["events"][-len(task["oracle_actions"]) :]
                            if len(events) != len(task["oracle_actions"]):
                                raise RuntimeError("Incomplete teacher trace")
                            history = []
                            rows = []
                            for i, event in enumerate(events):
                                # Use the same public graph shape as MultiCADEnv;
                                # native FreeCAD snapshot fields differ from CQ/bpy.
                                _, objects = resolve(
                                    task["initial_actions"]
                                    + task["oracle_actions"][:i],
                                    allow_empty=True,
                                    return_objects=True,
                                )
                                before = {
                                    "objects": objects,
                                    "geometry": event["observation_before"].get(
                                        "geometry"
                                    ),
                                }
                                if i:
                                    mesh = (
                                        case
                                        / "teacher"
                                        / events[i - 1]["checkpoint_mesh"]
                                    )
                                elif task["initial_actions"]:
                                    if engine == "freecad":
                                        mesh = case / "teacher/initial_mesh.json"
                                    else:
                                        mesh = (
                                            case
                                            / "teacher"
                                            / result["events"][-len(events) - 1][
                                                "checkpoint_mesh"
                                            ]
                                        )
                                else:
                                    mesh = case / "empty.json"
                                    mesh.write_text('{"vertices":[],"faces":[]}')
                                current = Path(
                                    render_mesh(
                                        mesh,
                                        case / "observations" / str(i),
                                        size=384,
                                        views=("iso",),
                                    )[0]
                                )
                                images = [
                                    str(p.relative_to(out)) for p in [*goals, current]
                                ]
                                rows.append(
                                    {
                                        "task_id": identifier,
                                        "step_id": i,
                                        "split_group": task["split_group"],
                                        "engine": engine,
                                        "observation_kind": "current_state",
                                        "images": images,
                                        "prompt": messages(
                                            engine, task, before, history, images
                                        ),
                                        "completion": [
                                            {
                                                "role": "assistant",
                                                "content": json.dumps(event["action"]),
                                            }
                                        ],
                                    }
                                )
                                history.append(
                                    {
                                        "action": event["action"],
                                        "response": json.dumps(event["action"]),
                                        "execution_ok": True,
                                    }
                                )
                            exports[task["split"]].extend(rows)
                            accepted = True
                        except Exception as exc:
                            if isinstance(
                                exc, RuntimeError
                            ) and "source changed" in str(exc):
                                raise
                            accepted = False
                            rejected.append({"task_id": identifier, "error": str(exc)})
                        manifest.append(
                            {
                                "task_id": identifier,
                                "engine": engine,
                                "family": family,
                                "mode": mode,
                                "input_modality": modality,
                                "split": task["split"],
                                "split_group": task["split_group"],
                                "accepted": accepted,
                            }
                        )
    write_jsonl(out / "manifest.jsonl", manifest)
    for split, rows in exports.items():
        write_jsonl(out / "exports" / f"{split}.feedback.jsonl", rows)
    # Snapshot the exact implementation used to produce positive labels.
    for folder in ("agent", "multicad"):
        for source in (Path(__file__).resolve().parents[1] / folder).glob("*.py"):
            target = out / "source_snapshot" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    summary = {
        "schema_version": "multicad.v1",
        "benchmark": "multicad-mechanical-v1",
        "engines": list(engines),
        "accepted": sum(r["accepted"] for r in manifest),
        "rejected": rejected,
        "split_counts": {
            s: sum(r["accepted"] and r["split"] == s for r in manifest) for s in exports
        },
        "feedback_counts": {s: len(r) for s, r in exports.items()},
        "supported_training_exports": ["train.feedback.jsonl"],
        "limitations": [
            "Restricted shared XY tool subset; no arbitrary Python",
            "Original synthetic families; not the official CADBench or CADWorld",
            "Views and dimensions are synthetic; no PBR/noisy-mesh tasks yet",
        ],
        "files": {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file()
            and ".runtime" not in p.parts
            and p.suffix in {".json", ".jsonl", ".png", ".py", ".FCStd", ".step"}
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def audit(root):
    root = Path(root).resolve()
    summary = json.loads((root / "summary.json").read_text())
    errors = []
    if summary.get("schema_version") != "multicad.v1":
        raise ValueError("Not a multi-engine dataset")
    for relative, digest in summary["files"].items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != digest:
            errors.append("Missing/changed file: " + relative)
    groups = {}
    accepted = {}
    tasks = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if groups.setdefault(row["split_group"], row["split"]) != row["split"]:
            errors.append("Cross-engine/variant split leakage")
        if row["accepted"]:
            accepted[row["task_id"]] = row
            task = json.loads(
                (root / "episodes" / row["task_id"] / "task.json").read_text()
            )
            tasks[row["task_id"]] = task
            if not task.get("goal_images"):
                errors.append("Accepted visual task has no goal assets")
            expected = make_task(
                task["family"],
                task["seed"],
                task["mode"],
                task["input_modality"],
                varied_edits=task.get("curriculum_version") == "varied-edits-v2",
            )
            if any(task.get(k) != v for k, v in expected.items()):
                errors.append("Task differs from deterministic specification")
            for label in ("reference", "teacher"):
                result = json.loads(
                    (
                        root / "episodes" / row["task_id"] / label / "result.json"
                    ).read_text()
                )
                if not result.get("execution_ok") or not result.get(
                    "validation", {}
                ).get("task_success"):
                    errors.append("Accepted native trace not successful")
    for split in ("train", "validation", "test"):
        for line in (
            (root / "exports" / f"{split}.feedback.jsonl").read_text().splitlines()
        ):
            row = json.loads(line)
            entry = accepted.get(row["task_id"], {})
            if entry.get("split") != split or entry.get("engine") != row["engine"]:
                errors.append("Wrong split/engine in export")
            task = tasks.get(row["task_id"])
            if task:
                step = row.get("step_id")
                if not isinstance(step, int) or not 0 <= step < len(
                    task["oracle_actions"]
                ):
                    errors.append("Invalid teacher step")
                elif (
                    json.loads(row["completion"][0]["content"])
                    != task["oracle_actions"][step]
                ):
                    errors.append("Completion differs from validated teacher")
                if row["images"][: len(task.get("goal_images", []))] != task.get(
                    "goal_images"
                ):
                    errors.append("Export lost or replaced goal images")
            blocks = [
                b
                for m in row["prompt"]
                if isinstance(m["content"], list)
                for b in m["content"]
                if b.get("type") == "image"
            ]
            if len(blocks) != len(row["images"]):
                errors.append("Image placeholders do not match assets")
            for relative in row["images"]:
                p = (root / relative).resolve()
                if not p.is_relative_to(root) or not p.is_file():
                    errors.append("Image escaped or missing")
    return {
        "ok": not errors,
        "errors": errors,
        "parent_groups": len(groups),
        "accepted": len(accepted),
        "schema_version": "multicad.v1",
    }
