"""Framework-neutral replay and public feedback for a multi-turn GRPO adapter."""

import json
from pathlib import Path

from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages, policy_images

from .factory import jsonl
from .task_identity import load_private_task


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


def replay_conversation(
    task_dir,
    task_id,
    messages,
    output_dir,
    *,
    backend=None,
    visual=True,
    require_train=True,
):
    task = load_private_task(task_dir, task_id, require_train=require_train)
    env = CADAgentEnv(
        output_dir, backend, visual=visual, asset_root=Path(task_dir).resolve().parent
    )
    observation, _ = env.reset(task)
    responses = [
        message_text(m["content"])
        for m in messages
        if m["role"] == "assistant" and m.get("content")
    ]
    reward = 0.0
    info = {}
    invalid = 0
    extra = False
    for text in responses:
        if env.closed:
            extra = True
            break
        try:
            action = json.loads(text)
        except ValueError:
            action = {"op": "invalid_json"}
        observation, reward, _done, _truncated, info = env.step(action)
        if info.get("error_type") in (
            "WorkerCrash",
            "ProtocolError",
            "AttributeError",
            "NameError",
            "ImportError",
        ):
            raise RuntimeError("Native infrastructure failed during rollout")
        invalid += int(not info["execution_ok"])
    finished = bool(
        responses
        and env.closed
        and env.actions
        and env.actions[-1]["op"] == "finish"
        and not extra
    )
    return {
        "observation": observation,
        "reward": reward if finished else 0.0,
        "finished": finished,
        "task_success": finished
        and bool(info.get("validation", {}).get("task_success")),
        "invalid_actions": invalid,
        "steps": env.attempts,
        "trajectory": str(env.episode_dir / "trajectory.jsonl"),
    }


def export_swift_tasks(dataset_dir, output_path):
    from .audit import audit_dataset

    root = Path(dataset_dir).resolve()
    if not audit_dataset(root)["ok"]:
        raise ValueError("Dataset audit failed")
    rows = []
    summary = json.loads((root / "summary.json").read_text())
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"] or entry["split"] != "train":
            continue
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        native = json.loads(
            (root / "episodes" / entry["task_id"] / "oracle/result.json").read_text()
        )
        observation = {
            "state": native["initial_observation"],
            "images": [],
            "goal_images": [],
        }
        for relative in task.get("goal_images", []):
            image = (root / relative).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError("Goal image must be inside the dataset")
            observation["goal_images"].append(str(image))
        if not task.get("goal_only_observation"):
            image = root / "episodes" / entry["task_id"] / "images/initial-iso.png"
            if not image.is_file():
                raise ValueError("Dataset needs current initial-state renders")
            observation["images"].append(str(image))
        images = policy_images(observation)
        if not images:
            raise ValueError("Visual rollout export requires an image")
        messages = agent_messages(task, observation, visual=True)
        # ms-swift accepts <image> placeholders with paths in its separate images column.
        for message in messages:
            if isinstance(message["content"], list):
                message["content"] = "".join(
                    "<image>" if block["type"] == "image" else block["text"]
                    for block in message["content"]
                )
        rows.append(
            {
                "task_id": task["task_id"],
                "messages": messages,
                "images": images,
                "split_group": task["split_group"],
            }
        )
    jsonl(output_path, rows)
    return {
        "rows": len(rows),
        "source_schema": summary["schema_version"],
        "contract": "one JSON action per turn; fresh environment per trajectory; private terminal native reward",
    }
