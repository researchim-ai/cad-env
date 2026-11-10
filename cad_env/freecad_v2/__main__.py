"""Generate native-verified v2 tasks, goal renders and causal SFT next actions."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from cad_env.agent.rendering import render_mesh
from cad_env.learning.audit import audit_dataset

from .backend import Backend
from .environment import Environment, messages
from .tasks import FAMILIES, make_task


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(
    output,
    per_family=1,
    seed=0,
    families=FAMILIES,
    *,
    task_factory=make_task,
    drawings=False,
):
    if per_family < 1 or not families or len(set(families)) != len(families):
        raise ValueError("Positive unique family budgets required")
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    backend = Backend()
    exports = {split: [] for split in ("train", "validation", "test")}
    manifest, seen = [], set()
    for family in families:
        for offset in range(per_family):
            task = task_factory(family, seed + offset)
            if task["split_group"] in seen:
                raise ValueError(
                    "Duplicate generated geometry lineage; choose different seeds"
                )
            seen.add(task["split_group"])
            case = root / "episodes" / task["task_id"]
            case.mkdir(parents=True)
            (case / "task.json").write_text(json.dumps(task, indent=2))
            correct = backend.run(
                task["actions"],
                case / "oracle",
                reference=task["actions"],
                expected_volume=task["expected_volume"],
            )
            # Every generated task also has a deliberately wrong, but valid,
            # translated candidate. A self-comparison alone cannot audit reward.
            wrong = copy.deepcopy(task["actions"][:-1])
            wrong += [
                {
                    "op": "translate",
                    "name": "NegativeControl",
                    "source": task["actions"][-1]["targets"][0],
                    "vector": [1, 0, 0],
                },
                {"op": "finish", "targets": ["NegativeControl"]},
            ]
            negative = backend.run(wrong, case / "negative", reference=task["actions"])
            accepted = bool(
                correct.get("execution_ok")
                and correct.get("validation", {}).get("task_success")
                and negative.get("execution_ok")
                and not negative.get("validation", {}).get("task_success")
            )
            row = {
                "task_id": task["task_id"],
                "split_group": task["split_group"],
                "split": task["split"],
                "family": family,
                "accepted": accepted,
                "error": correct.get("error"),
                "negative_control_rejected": negative.get("execution_ok")
                and not negative.get("validation", {}).get("task_success"),
            }
            if accepted:
                paths = render_mesh(
                    case / "oracle/mesh.json",
                    case / "goal",
                    size=384,
                    views=("iso", "top", "front"),
                )
                # render_mesh returns a list of paths (not a specification transcription).
                images = [str(Path(p).relative_to(root)) for p in paths]
                image_groups = {"renders": images}
                if drawings:
                    from .drawings import export_pair

                    paired = export_pair(
                        case / "oracle/final.step", task, case / "drawings"
                    )
                    image_groups = {
                        standard: [
                            str(Path(p).relative_to(root)) for p in rep["images"]
                        ]
                        for standard, rep in paired.items()
                    }
                    row["drawing_profiles"] = {
                        s: str(Path(rep["root"]).relative_to(root))
                        for s, rep in paired.items()
                    }
                env = Environment(case / "rollout", backend=backend)
                observation = {
                    "state": [],
                    "attempts": 0,
                    "remaining_actions": 128,
                    "error": None,
                }
                feedback = []
                for step, action in enumerate(task["actions"]):
                    samples = []
                    for standard, images in image_groups.items():
                        prompt = messages(
                            task["instruction"],
                            observation,
                            env.history,
                            image_count=len(images),
                        )
                        samples.append(
                            {
                                "task_id": task["task_id"],
                                "split_group": task["split_group"],
                                "step_id": step,
                                "prompt": prompt,
                                "completion": [
                                    {
                                        "role": "assistant",
                                        "content": [
                                            {"type": "text", "text": json.dumps(action)}
                                        ],
                                    }
                                ],
                                "images": images,
                                "standard": standard,
                            }
                        )
                    observation, reward, done, info = env.step(
                        action,
                        reference=task["actions"] if action["op"] == "finish" else None,
                    )
                    if not info["execution_ok"] or (
                        step == len(task["actions"]) - 1 and (not done or reward != 1)
                    ):
                        raise RuntimeError(
                            "Teacher prefix/terminal verification failed: " + str(info)
                        )
                    feedback.extend(samples)
                exports[task["split"]].extend(feedback)
                row["artifact_sha256"] = digest(case / "oracle/output.FCStd")
            manifest.append(row)
            print(f"{family} {task['task_id']} accepted={accepted}", flush=True)
    (root / "exports").mkdir()
    for split, rows in exports.items():
        (root / "exports" / f"{split}.feedback.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest)
    )
    for split in ("validation", "test"):
        (root / f"{split}-selection.json").write_text(
            json.dumps(
                sorted(
                    [r for r in manifest if r["split"] == split],
                    key=lambda r: r["task_id"],
                ),
                indent=2,
            )
        )
    package_root = Path(__file__).absolute().parents[2]
    for source, expected in backend.sources.items():
        source = Path(source)
        if digest(source) != expected:
            raise RuntimeError("Source changed before dataset snapshot")
        target = root / "source_snapshot" / source.relative_to(package_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    hashes = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix
        in (".png", ".FCStd", ".json", ".jsonl", ".py", ".step", ".svg", ".pdf")
        and ".runtime" not in p.parts
    }
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "benchmark": "freecad-expanded-v2",
        "action_contract": "single_action",
        "cad_contract": "freecad.v2",
        "supported_training_exports": ["train.feedback.jsonl"],
        "attempted": len(manifest),
        "accepted": sum(r["accepted"] for r in manifest),
        "counts": {s + ".feedback": len(rows) for s, rows in exports.items()},
        "artifact_sha256": hashes,
        "export_sha256": {},
        "source_hashes": backend.sources,
        "drawing_profiles": ["iso", "gost"] if drawings else [],
        "scope": "Native verified clean-prefix teachers. Experimental ISO/GOST profiles with dimensional appendices."
        if drawings
        else "Native operation probes and clean-prefix teachers; goal renders plus dimensioned text, not ISO/GOST sheets. No model skill or full FreeCAD coverage claim.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = audit_dataset(root)
    (root / "audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["ok"]:
        raise ValueError(audit["errors"])
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--per-family", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    args = p.parse_args()
    print(
        json.dumps(
            generate(args.output, args.per_family, args.seed, args.families), indent=2
        )
    )


if __name__ == "__main__":
    main()
