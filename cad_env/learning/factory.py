"""Verified procedural dataset factory and leakage-safe training exports."""

import copy
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cad_env.agent import FAMILIES, MODES, FreeCADBackend, make_task
from cad_env.agent.observations import agent_messages, transition
from cad_env.agent.rendering import render_mesh
from cad_env.agent.tasks import allowed_edits, expected_metrics, prompt_messages

from .rewards import score_completion


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def teacher_transition(action, info, observation):
    # Deployed policies retain both parsed action and raw assistant response.
    # Teacher responses use the exact canonical JSON used for their completion.
    return {
        **transition(action, info, observation),
        "response": json.dumps(action)[:2000],
    }


def negative_actions(task):
    actions = copy.deepcopy(task["oracle_actions"])
    if task["mode"] != "create":
        actions[0]["value"] *= 1.4
    else:
        for action in actions:
            if action["op"] == "pad":
                action["length"] *= 1.4
                break
    return actions


def generate_dataset(
    output_dir,
    *,
    seeds=10,
    start_seed=0,
    families=FAMILIES,
    modes=MODES,
    executable=None,
    timeout=30,
    workers=1,
    render=True,
    preferences=False,
    holdout_families=(),
    task_specs=None,
):
    if seeds < 1 or workers < 1 or start_seed < 0:
        raise ValueError(
            "seeds/workers must be positive, start_seed must be nonnegative"
        )
    if (
        not families
        or not modes
        or len(set(families)) != len(families)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("families and modes must be non-empty and unique")
    tasks = (
        copy.deepcopy(task_specs)
        if task_specs is not None
        else [
            make_task(f, seed, mode)
            for seed in range(start_seed, start_seed + seeds)
            for f in families
            for mode in modes
        ]
    )
    if not tasks:
        raise ValueError("At least one task is required")
    has_visual_goals = any(t.get("goal_spec") or t.get("goal_images") for t in tasks)
    if has_visual_goals and not render:
        raise ValueError("Visual-goal tasks require rendered feedback exports")
    if has_visual_goals and preferences:
        raise ValueError(
            "Visual-goal curricula require image-aware rollout preference exporters"
        )
    if set(holdout_families) - set(families):
        raise ValueError("Holdout families must be included in families")
    unique = {}
    for task in tasks:
        if task["family"] in holdout_families:
            task["split"] = "test"
        key = json.dumps(
            [
                task["family"],
                task["mode"],
                task["reference_actions"],
                task["initial_actions"],
                task.get("input_modality", "default"),
                # Different visual conditions can share the same final CAD.
                # Keep them as observations, while split_group still counts
                # their common physical lineage only once.
                task.get("goal_spec"),
                task["instruction"],
            ],
            sort_keys=True,
        )
        unique.setdefault(key, task)
    duplicate_skipped = len(tasks) - len(unique)
    tasks = list(unique.values())
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Output directory is not empty; use a new directory to preserve provenance"
        )
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for folder in ("agent", "learning"):
        for source in sorted((source_root / folder).glob("*.py")):
            relative = source.relative_to(source_root)
            target = root / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_hashes[str(relative)] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    backend = FreeCADBackend(executable, timeout)
    started = time.monotonic()

    def generate(task):
        episode = root / "episodes" / task["task_id"]
        episode.mkdir(parents=True)
        if task.get("goal_spec"):
            from .visual_tasks import render_goal

            task["goal_images"] = [str((episode / "goal.png").relative_to(root))]
            render_goal(task, episode / "goal.png")
        (episode / "task.json").write_text(
            json.dumps(task, indent=2, ensure_ascii=False)
        )
        result = backend.run(
            task["oracle_actions"],
            episode / "oracle",
            initial_actions=task["initial_actions"],
            reference_actions=task["reference_actions"],
            record=True,
            mesh=render and not task.get("goal_only_observation"),
            tolerance=task["tolerance"],
            expected_metrics=expected_metrics(task),
            allowed_edits=allowed_edits(task),
            reward_mode=task.get("reward_mode", "dense_geometry"),
        )
        accepted = bool(
            result["execution_ok"] and result.get("validation", {}).get("task_success")
        )
        if accepted:
            jsonl(episode / "trajectory.jsonl", result["events"])
        rejected = None
        if accepted and preferences:
            wrong = negative_actions(task)
            score = score_completion(
                task, json.dumps(wrong), episode / "negative", backend
            )
            if (
                not score["task_success"]
                and score["reward"] < result["validation"]["reward"]
            ):
                rejected = wrong
        return {
            "task": task,
            "result": result,
            "accepted": accepted,
            "episode": episode,
            "rejected": rejected,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        episodes = list(pool.map(generate, tasks))
    exports = {
        (split, kind): []
        for split in ("train", "validation", "test")
        for kind in ("sft", "steps", "grpo", "vlm", "feedback", "preference")
    }
    manifest = []
    for item in episodes:
        task, result, episode = item["task"], item["result"], item["episode"]
        manifest.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "mode": task["mode"],
                "split": task["split"],
                "split_group": task["split_group"],
                "accepted": item["accepted"],
                "execution_seconds": result["execution_seconds"],
                "validation": result.get("validation"),
                "error": result.get("error"),
                "artifact_sha256": result.get("artifact_sha256"),
                **(
                    {
                        "condition_group": task["condition_group"],
                        "evaluation_tier": task.get("evaluation_tier", "iid"),
                    }
                    if "condition_group" in task
                    else {}
                ),
            }
        )
        if not item["accepted"]:
            continue
        split = task["split"]
        prompt = prompt_messages(task, result["initial_observation"])
        completion = [
            {
                "role": "assistant",
                "content": json.dumps(task["oracle_actions"], ensure_ascii=False),
            }
        ]
        meta = {
            "task_id": task["task_id"],
            "split_group": task["split_group"],
            "family": task["family"],
            "mode": task["mode"],
        }
        # Unconditioned text exports cannot identify a target encoded only in
        # image pixels. Keep them empty for visual-goal tasks rather than emit
        # contradictory training examples.
        if not task.get("goal_images"):
            exports[split, "sft"].append(
                {**meta, "prompt": prompt, "completion": completion}
            )
            exports[split, "grpo"].append({**meta, "prompt": prompt})
            history = []
            for event in result["events"]:
                step_prompt = prompt_messages(task, event["observation_before"])
                step_prompt[0]["content"] += (
                    "\nFor this next-action task return ONE JSON action object, not an array."
                )
                step_prompt[1]["content"] += (
                    "\nPreviously accepted actions: " + json.dumps(history)
                )
                exports[split, "steps"].append(
                    {
                        **meta,
                        "step_id": event["step_id"],
                        "prompt": step_prompt,
                        "completion": [
                            {
                                "role": "assistant",
                                "content": json.dumps(event["action"]),
                            }
                        ],
                    }
                )
                history.append(event["action"])
        if render and not task.get("goal_only_observation"):
            images = render_mesh(episode / "oracle" / "mesh.json", episode / "images")
            if (episode / "oracle" / "initial_mesh.json").is_file():
                render_mesh(
                    episode / "oracle" / "initial_mesh.json",
                    episode / "images",
                    prefix="initial",
                )
            visual_prompt = copy.deepcopy(prompt)
            visual_prompt[-1]["content"] = [
                {
                    "type": "text",
                    "text": prompt[-1]["content"]
                    + "\nThe images show the desired final geometry (iso, top, front, right). Dimensions are specified in the task.",
                }
            ] + [{"type": "image"} for _ in images]
            visual_completion = copy.deepcopy(completion)
            for message in visual_prompt + visual_completion:
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
            exports[split, "vlm"].append(
                {
                    **meta,
                    "prompt": visual_prompt,
                    "completion": visual_completion,
                    "images": [str(Path(p).relative_to(root)) for p in images],
                    "render": {
                        "projection": "orthographic",
                        "views": ["iso", "top", "front", "right"],
                        "size_px": 384,
                        "renderer": "numpy-zbuffer-v1",
                    },
                }
            )
        if render:
            feedback_history = []
            rejected_observation = None
            if task["mode"] == "repair":
                bad_action = {
                    "op": "set_parameter",
                    "target": "MissingObject",
                    "parameter": "length",
                    "value": 1,
                }
                rejected_result = backend.run(
                    [bad_action],
                    episode / "rejected_action",
                    initial_actions=task["initial_actions"],
                )
                if rejected_result["execution_ok"]:
                    raise RuntimeError(
                        "Recovery teacher's invalid action unexpectedly succeeded"
                    )
                rejected_observation = rejected_result["error"]
                feedback_history.append(
                    teacher_transition(
                        bad_action,
                        {"execution_ok": False},
                        {"error": rejected_observation},
                    )
                )
            for index, event in enumerate(result["events"]):
                current_images = []
                if not task.get("goal_only_observation"):
                    mesh_relative = (
                        "initial_mesh.json"
                        if index == 0
                        else result["events"][index - 1]["checkpoint_mesh"]
                    )
                    current_images = render_mesh(
                        episode / "oracle" / mesh_relative,
                        episode / "feedback_images",
                        prefix=f"state-{index:03d}",
                        size=256,
                        views=("iso",),
                    )
                observation = {
                    "state": event["observation_before"],
                    "remaining_steps": task["max_steps"] - index,
                    "images": current_images,
                    "goal_images": [str(root / p) for p in task.get("goal_images", [])],
                }
                if index == 0 and rejected_observation:
                    observation["error"] = rejected_observation
                feedback_prompt = agent_messages(
                    task, observation, feedback_history, visual=True
                )
                exports[split, "feedback"].append(
                    {
                        **meta,
                        "step_id": index,
                        "observation_kind": "current_state",
                        "recovery": bool(index == 0 and rejected_observation),
                        "goal_required": bool(task.get("goal_images")),
                        "prompt": feedback_prompt,
                        "completion": [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(event["action"]),
                                    }
                                ],
                            }
                        ],
                        "images": [
                            str(Path(image).relative_to(root))
                            for image in observation["goal_images"] + current_images
                        ],
                    }
                )
                # A failed repeated sketch must teach progression, not another
                # duplicate command. Execute the failure against the exact
                # committed prefix; keep this as an independent variant.
                if task["mode"] == "create" and index == 2:
                    repeated_action = result["events"][index - 1]["action"]
                    if repeated_action["op"] in ("sketch_rectangle", "sketch_circles"):
                        rejected = backend.run(
                            [repeated_action],
                            episode / "rejected_duplicate",
                            initial_actions=task["initial_actions"]
                            + [e["action"] for e in result["events"][:index]],
                        )
                        if rejected[
                            "execution_ok"
                        ] or "Duplicate object name" not in rejected.get("error", ""):
                            raise RuntimeError(
                                "Duplicate recovery must produce a real duplicate-name error"
                            )
                        recovery_observation = {
                            **observation,
                            "error": rejected["error"],
                            "remaining_steps": observation["remaining_steps"] - 1,
                        }
                        variant = copy.deepcopy(exports[split, "feedback"][-1])
                        variant.update(
                            recovery=True,
                            recovery_kind="duplicate_name",
                            prompt=agent_messages(
                                task,
                                recovery_observation,
                                feedback_history
                                + [
                                    teacher_transition(
                                        repeated_action,
                                        {"execution_ok": False},
                                        recovery_observation,
                                    )
                                ],
                                visual=True,
                            ),
                        )
                        exports[split, "feedback"].append(variant)
                feedback_history.append(
                    teacher_transition(
                        event["action"],
                        {"execution_ok": True},
                        event["observation_after"],
                    )
                )
        if item["rejected"] is not None:
            exports[split, "preference"].append(
                {
                    **meta,
                    "prompt": prompt,
                    "chosen": completion,
                    "rejected": [
                        {"role": "assistant", "content": json.dumps(item["rejected"])}
                    ],
                }
            )
    for (split, kind), rows in exports.items():
        jsonl(root / "exports" / f"{split}.{kind}.jsonl", rows)
    jsonl(root / "manifest.jsonl", manifest)
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "exports").glob("*.jsonl"))
    }
    summary = {
        "schema_version": "cad-env.dataset.v1",
        "attempted": len(tasks),
        "accepted": sum(e["accepted"] for e in episodes),
        "rejected": sum(not e["accepted"] for e in episodes),
        "duplicate_skipped": duplicate_skipped,
        "elapsed_seconds": time.monotonic() - started,
        "seed_range": [start_seed, start_seed + seeds],
        "families": list(families),
        "modes": list(modes),
        "workers": workers,
        "counts": {f"{s}.{k}": len(rows) for (s, k), rows in exports.items()},
        "runtime": next(
            (
                e["result"].get("runtime")
                for e in episodes
                if e["result"].get("runtime")
            ),
            None,
        ),
        "worker_sha256": episodes[0]["result"].get("worker_sha256"),
        "source_sha256": source_hashes,
        "artifact_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("episodes", "source_snapshot")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file()
            and (
                p.suffix in (".png", ".FCStd", ".py")
                or p.name.endswith("-mesh.json")
                or p.name.endswith("-cameras.json")
                or p.name
                in (
                    "task.json",
                    "trajectory.jsonl",
                    "view-cameras.json",
                    "initial-cameras.json",
                    "initial_mesh.json",
                    "mesh.json",
                )
            )
        },
        "export_sha256": hashes,
        "holdout_families": list(holdout_families),
        "split_policy": "sha256(family + invariant base dimensions); edit/repair, views, and dimension variations share a split; optional entire-family test holdout",
    }
    if has_visual_goals:
        summary["supported_training_exports"] = ["train.feedback.jsonl"]
        summary["excluded_export_reason"] = (
            "Visual goals require images and the sequential action contract"
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
