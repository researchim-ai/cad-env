"""Framework-neutral replay and public feedback for a multi-turn GRPO adapter."""

import json
import re
from pathlib import Path

from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages

from .factory import jsonl


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
    if not isinstance(task_id, str) or not re.fullmatch(
        r"[a-z_]+-[0-9]+-(create|edit|repair)", task_id
    ):
        raise ValueError("Invalid task identity")
    task = json.loads((Path(task_dir) / task_id / "task.json").read_text())
    if require_train and task["split"] != "train":
        raise ValueError("RL rewards cannot consume held-out tasks")
    env = CADAgentEnv(output_dir, backend, visual=visual)
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
        observation, reward, done, truncated, info = env.step(action)
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
    root = Path(dataset_dir).resolve()
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
        image = root / "episodes" / entry["task_id"] / "images/initial-iso.png"
        if not image.is_file():
            raise ValueError(
                "Dataset needs current initial-state renders, including empty documents"
            )
        observation = {"state": native["initial_observation"], "images": [str(image)]}
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
                "images": [str(image)],
                "split_group": task["split_group"],
            }
        )
    jsonl(output_path, rows)
    return {
        "rows": len(rows),
        "source_schema": summary["schema_version"],
        "contract": "one JSON action per turn; fresh environment per trajectory; private terminal native reward",
    }
