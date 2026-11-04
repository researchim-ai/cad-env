"""Render exact model trajectories as inspectable native CAD examples.

Examples never replace all-case metrics. Replays use recorded accepted actions;
no teacher action or parameter correction is inserted.
"""

import argparse
import json
import shutil
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.rendering import render_mesh
from cad_env.agent.tasks import allowed_edits, expected_metrics

from .protocol import digest


def render_examples(dataset, evaluation, output):
    root, path, out = (Path(p).resolve() for p in (dataset, evaluation, output))
    report = json.loads(path.read_text())
    if report["status"] != "complete" or report.get("limited_evaluation"):
        raise ValueError("Examples require a complete evaluation")
    if report["dataset_digest"] != digest(
        json.loads((root / "summary.json").read_text())
    ):
        raise ValueError("Dataset differs from evaluated data")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new example directory")
    out.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for episode in report["episodes"]:
        task = json.loads(
            (root / "episodes" / episode["task_id"] / "task.json").read_text()
        )
        if task.get("input_modality", "image") != "image":
            continue
        chosen.setdefault(
            (task["family"], bool(episode["task_success"])), (task, episode)
        )
    backend = FreeCADBackend()
    rows = []
    for (family, success), (task, episode) in chosen.items():
        case = out / (family + ("-success" if success else "-failure"))
        case.mkdir()
        events = [
            json.loads(l)
            for l in (path.parent / episode["trajectory"]).read_text().splitlines()
        ]
        actions = [r["action"] for r in events if r["info"].get("execution_ok")]
        result = backend.run(
            actions,
            case / "native",
            mesh=True,
            initial_actions=task["initial_actions"],
            reference_actions=task["reference_actions"],
            allowed_edits=allowed_edits(task),
            expected_metrics=expected_metrics(task),
            tolerance=task["tolerance"],
            reward_mode=task.get("reward_mode", "dense_geometry"),
        )
        if (
            not result["execution_ok"]
            or result["runtime"] != report["protocol"]["runtime"]
        ):
            raise RuntimeError(
                "Recorded native trajectory did not replay under the same runtime"
            )
        native = Path(result["output_dir"])
        views = render_mesh(
            native / "mesh.json", case / "views", size=512, views=("iso", "top")
        )
        for i, goal in enumerate(task.get("goal_images", [])):
            source = (root / goal).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Goal escaped dataset")
            shutil.copyfile(source, case / f"goal-{i}.png")
        (case / "model-actions.json").write_text(json.dumps(actions, indent=2))
        rows.append(
            {
                "task_id": task["task_id"],
                "family": family,
                "original_task_success": success,
                "replayed_geometry_success": result.get("validation", {}).get(
                    "task_success", False
                ),
                "native_document": str((native / "output.FCStd").relative_to(out)),
                "views": [str(Path(p).relative_to(out)) for p in views],
                "source_trajectory": str(path.parent / episode["trajectory"]),
            }
        )
    result = {
        "status": "complete",
        "model_identity": report["model_identity"],
        "examples": rows,
        "selection": "First task ID per image-conditioned family and success/failure outcome; all-case evaluation remains authoritative",
        "note": "Exact accepted model actions only. Geometry replay cannot turn an original truncated/unfinished policy episode into a policy success.",
    }
    (out / "examples.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(
        json.dumps(
            render_examples(args.dataset, args.evaluation, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
