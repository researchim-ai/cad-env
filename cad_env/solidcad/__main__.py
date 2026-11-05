"""Generate and natively verify experimental advanced-feature task packages."""

import argparse
import hashlib
import json
from pathlib import Path

from cad_env.agent.rendering import render_mesh

from .backend import SolidBackend
from .schema import TOOLS
from .tasks import FAMILIES, make_task


def generate(output, engines, seeds=2, families=FAMILIES, python=None, blender=None):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh advanced corpus directory")
    root.mkdir(parents=True)
    rows = []
    exports = {s: [] for s in ["train", "validation", "test"]}
    backends = {
        engine: SolidBackend(engine, python=python, blender=blender)
        for engine in engines
    }
    guide = json.dumps(
        {
            op: {"required": sorted(spec[0]), "optional": sorted(spec[1])}
            for op, spec in TOOLS.items()
        }
    )
    for seed in range(seeds):
        for family in families:
            task = make_task(family, seed)
            for engine, backend in backends.items():
                name = (
                    "s-"
                    + hashlib.sha256(
                        (task["split_group"] + engine).encode()
                    ).hexdigest()[:24]
                )
                if any(row["task_id"] == name for row in rows):
                    continue
                case = root / "episodes" / name
                case.mkdir(parents=True)
                (case / "task.json").write_text(
                    json.dumps({**task, "engine": engine, "task_id": name}, indent=2)
                )
                result = backend.run(
                    task["program"],
                    case / "native",
                    reference=task["program"],
                    expected_volume=task["expected_volume"],
                )
                success = bool(
                    result.get("execution_ok")
                    and result.get("validation", {}).get("task_success")
                )
                if success and task["base_volume"] is not None:
                    success = 0 < result["geometry"]["volume"] < task["base_volume"]
                rows.append(
                    {
                        "task_id": name,
                        "engine": engine,
                        "family": family,
                        "split": task["split"],
                        "split_group": task["split_group"],
                        "accepted": success,
                        "error": result.get("error"),
                    }
                )
                if success:
                    views = render_mesh(
                        case / "native/mesh.json",
                        case / "goal",
                        size=384,
                        views=("iso", "top"),
                    )
                    images = [str(Path(v).relative_to(root)) for v in views]
                    prompt = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Use solidcad.v1 tools. Return only a JSON array of actions. Allowed tools: "
                                    + guide,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": task["instruction"]}]
                            + [{"type": "image"} for _ in images],
                        },
                    ]
                    exports[task["split"]].append(
                        {
                            "task_id": name,
                            "engine": engine,
                            "split_group": task["split_group"],
                            "images": images,
                            "prompt": prompt,
                            "completion": [
                                {
                                    "role": "assistant",
                                    "content": json.dumps(task["program"]),
                                }
                            ],
                        }
                    )
                (root / "progress.json").write_text(
                    json.dumps(
                        {
                            "cases": len(rows),
                            "accepted": sum(r["accepted"] for r in rows),
                        }
                    )
                )
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for split, data in exports.items():
        directory = root / "exports"
        directory.mkdir(exist_ok=True)
        (directory / (split + ".program.jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in data)
        )
    report = {
        "status": "complete",
        "schema_version": "solidcad.v1",
        "accepted": sum(r["accepted"] for r in rows),
        "failed": [r for r in rows if not r["accepted"]],
        "action_contract": "whole_program",
        "scope": "Advanced native feature probes and draft program SFT. Separate from frozen portable single-action training. Not an official external benchmark.",
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--engines",
        nargs="+",
        choices=["freecad", "cadquery", "blender"],
        default=["freecad", "cadquery", "blender"],
    )
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--python")
    p.add_argument("--blender")
    args = vars(p.parse_args())
    if args["seeds"] < 1:
        raise ValueError("Positive seed count required")
    report = generate(**args)
    print(json.dumps(report, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
