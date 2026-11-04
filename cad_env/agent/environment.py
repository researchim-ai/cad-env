"""Replay-based sequential environment: rejected actions never corrupt state."""

import json
import uuid
from pathlib import Path

from .backend import FreeCADBackend
from .schema import validate_action
from .tasks import allowed_edits, expected_metrics, public_task


class CADAgentEnv:
    def __init__(
        self, work_dir, backend=None, *, visual=False, views=("iso",), asset_root=None
    ):
        self.work_dir = Path(work_dir).resolve()
        self.backend = backend or FreeCADBackend()
        self.visual = visual
        self.views = tuple(views)
        self.asset_root = Path(asset_root).resolve() if asset_root else None
        self.task = None
        self.closed = True

    def reset(self, task):
        self.task = task
        self.actions = []
        self.attempts = 0
        self.closed = False
        self.episode_dir = self.work_dir / (
            task["task_id"] + "-" + uuid.uuid4().hex[:10]
        )
        self.episode_dir.mkdir(parents=True)
        result = self.backend.run(
            [],
            self.episode_dir / "reset",
            initial_actions=task["initial_actions"],
            mesh=self.visual and not task.get("goal_only_observation"),
        )
        if not result["execution_ok"]:
            self.closed = True
            raise RuntimeError(
                "Invalid task initial state: " + result.get("error", "unknown error")
            )
        self.observation = self._observation(result)
        return self.observation, {
            "episode_dir": str(self.episode_dir),
            "task": public_task(task),
        }

    def _observation(self, result):
        observation = {
            "task": public_task(self.task),
            "state": result.get("observation", {}),
            "remaining_steps": max(0, self.task["max_steps"] - self.attempts),
        }
        if self.visual:
            from .rendering import render_mesh

            folder = Path(result["output_dir"])
            observation["images"] = (
                []
                if self.task.get("goal_only_observation")
                else render_mesh(
                    folder / "mesh.json",
                    folder / "public_views",
                    size=256,
                    views=self.views,
                )
            )
            observation["goal_images"] = []
            for relative in self.task.get("goal_images", []):
                if self.asset_root is None:
                    raise ValueError("Visual goal tasks require an asset_root")
                path = (self.asset_root / relative).resolve()
                if not path.is_relative_to(self.asset_root) or not path.is_file():
                    raise ValueError("Goal image must exist inside the dataset")
                observation["goal_images"].append(str(path))
        return observation

    def step(self, action):
        if self.task is None or self.closed:
            raise RuntimeError(
                "Call reset before step; episode may already be finished"
            )
        self.attempts += 1
        before = self.observation
        reward = 0.0
        info = {}
        finished = False
        try:
            action = validate_action(action)
            finished = action["op"] == "finish"
            result = self.backend.run(
                self.actions + [action],
                self.episode_dir / f"step-{self.attempts:03d}",
                initial_actions=self.task["initial_actions"],
                reference_actions=self.task["reference_actions"] if finished else None,
                allowed_edits=allowed_edits(self.task) if finished else None,
                reward_mode=self.task.get("reward_mode", "dense_geometry"),
                mesh=self.visual and not self.task.get("goal_only_observation"),
                tolerance=self.task["tolerance"],
                expected_metrics=expected_metrics(self.task) if finished else None,
            )
            info = {
                "execution_ok": result["execution_ok"],
                "result_dir": result["output_dir"],
            }
            if result["execution_ok"]:
                self.actions.append(action)
                self.observation = self._observation(result)
                if finished:
                    validation = result.get("validation", {})
                    if (
                        validation.get("evaluator_error")
                        or result.get("error_origin") == "evaluation"
                    ):
                        raise RuntimeError(
                            "Evaluator failure: " + str(result.get("error"))
                        )
                    reward = validation.get("reward", 0.0)
                    info["validation"] = validation
            else:
                self.observation = {
                    **self.observation,
                    "remaining_steps": max(0, self.task["max_steps"] - self.attempts),
                    "error": result.get("error", "FreeCAD worker failed"),
                }
                info["error_type"] = result.get("error_type")
                finished = False
        except ValueError as exc:
            self.observation = {
                **self.observation,
                "remaining_steps": max(0, self.task["max_steps"] - self.attempts),
                "error": str(exc),
            }
            info = {"execution_ok": False, "error_type": "InvalidAction"}
        truncated = self.attempts >= self.task["max_steps"] and not finished
        self.closed = finished or truncated
        try:
            logged_action = json.loads(json.dumps(action, allow_nan=False))
        except (ValueError, TypeError):
            logged_action = {"invalid_input_repr": repr(action)[:2000]}
        event = {
            "step_id": self.attempts,
            "observation_before": before,
            "action": logged_action,
            "observation_after": self.observation,
            "reward": reward,
            "terminated": finished,
            "truncated": truncated,
            "info": info,
        }
        with (self.episode_dir / "trajectory.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        return self.observation, reward, finished, truncated, info
