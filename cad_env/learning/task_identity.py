"""Private task lookup shared by reward and replay adapters."""

import json
import re
from pathlib import Path


def load_private_task(task_dir, task_id, *, require_train=False):
    if not isinstance(task_id, str) or not re.fullmatch(
        r"(?:[a-z_]+-[0-9]+-(?:create|edit|repair)|[vc]-[0-9a-f]{24})", task_id
    ):
        raise ValueError("Invalid task identity")
    root = Path(task_dir).resolve()
    path = (root / task_id / "task.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Task escaped the private registry")
    task = json.loads(path.read_text())
    if task["task_id"] != task_id:
        raise ValueError("Task registry identity mismatch")
    if require_train and task["split"] != "train":
        raise ValueError("RL rewards cannot consume held-out tasks")
    return task
