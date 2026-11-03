"""Execution reward usable by TRL GRPOTrainer and offline rollout collectors."""

import json
import math
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.schema import validate_program
from cad_env.agent.tasks import allowed_edits, expected_metrics


def parse_completion(completion):
    if isinstance(completion, list):
        if len(completion) != 1 or completion[0].get("role") != "assistant":
            raise ValueError("Expected one assistant completion")
        completion = completion[0].get("content")
    if not isinstance(completion, str) or len(completion) > 100000:
        raise ValueError("Expected a bounded JSON text completion")
    text = completion.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Invalid JSON code fence")
        text = match.group(1)
    data = json.loads(text)
    if isinstance(data, dict):
        if set(data) != {"actions"}:
            raise ValueError("Only actions may appear in the completion object")
        data = data["actions"]
    actions = validate_program(data)
    if not actions or actions[-1]["op"] != "finish":
        raise ValueError("A complete program must end with finish")
    return actions


def score_completion(task, completion, output_dir, backend=None):
    try:
        actions = parse_completion(completion)
    except (ValueError, TypeError, KeyError) as exc:
        return {
            "reward": 0.0,
            "task_success": False,
            "execution_ok": False,
            "error_type": "InvalidCompletion",
            "error": str(exc),
        }
    backend = backend or FreeCADBackend()
    result = backend.run(
        actions,
        output_dir,
        initial_actions=task["initial_actions"],
        reference_actions=task["reference_actions"],
        tolerance=task["tolerance"],
        expected_metrics=expected_metrics(task),
        allowed_edits=allowed_edits(task),
    )
    if result.get("error_origin") == "evaluation" or result.get("validation", {}).get(
        "evaluator_error"
    ):
        raise RuntimeError("Trusted evaluator failed: " + str(result.get("error")))
    if result.get("error_type") in (
        "WorkerCrash",
        "ProtocolError",
        "AttributeError",
        "NameError",
        "ImportError",
        "ModuleNotFoundError",
    ):
        raise RuntimeError("FreeCAD infrastructure failed: " + str(result.get("error")))
    validation = result.get("validation", {})
    return {
        "reward": float(validation.get("reward", 0.0))
        if result["execution_ok"]
        else 0.0,
        "task_success": bool(validation.get("task_success", False))
        and result["execution_ok"],
        "execution_ok": result["execution_ok"],
        "validation": validation,
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "result_dir": result["output_dir"],
        "execution_seconds": result["execution_seconds"],
    }


class CADReward:
    """TRL custom reward. Dataset task_id columns route to private task manifests.

    No reference programs are included in model prompts. The callable deliberately
    raises on infrastructure/evaluator failures instead of silently learning zero.
    """

    __name__ = "freecad_geometry_reward"

    def __init__(
        self, task_dir, rollout_dir, *, executable=None, timeout=30, workers=1
    ):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.task_dir = Path(task_dir).resolve()
        self.rollout_dir = Path(rollout_dir).resolve()
        self.backend = FreeCADBackend(executable, timeout)
        self.workers = workers

    def score(self, task_id, completion):
        if not isinstance(task_id, str) or not re.fullmatch(
            r"[a-z_]+-[0-9]+-(create|edit|repair)", task_id
        ):
            raise ValueError("Invalid task_id")
        task = json.loads((self.task_dir / task_id / "task.json").read_text())
        out = self.rollout_dir / task_id / uuid.uuid4().hex
        out.mkdir(parents=True)
        score = score_completion(task, completion, out, self.backend)
        (out / "score.json").write_text(json.dumps(score, indent=2))
        (out / "completion.json").write_text(json.dumps(completion, ensure_ascii=False))
        return score

    def __call__(self, completions, task_id, **kwargs):
        if len(completions) != len(task_id):
            raise ValueError("Each completion must have a task_id")
        pairs = list(zip(task_id, completions))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            scores = list(pool.map(lambda pair: self.score(*pair), pairs))
        return [s["reward"] for s in scores]


def group_statistics(rewards, epsilon=1e-8):
    """Diagnostics only; GRPOTrainer owns policy loss and advantage normalization."""
    if len(rewards) < 2 or any(not math.isfinite(r) for r in rewards):
        raise ValueError("A group needs at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return {
        "mean": mean,
        "std": std,
        "zero_variance": std <= epsilon,
        "normalized_advantages": [(r - mean) / (std + epsilon) for r in rewards],
    }
