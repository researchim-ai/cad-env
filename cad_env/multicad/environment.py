"""Sequential tool environment for the portable multi-engine action subset."""

import copy
import json
from pathlib import Path

from cad_env.agent.rendering import render_mesh
from cad_env.agent.schema import validate_action

from .backend import MultiCADBackend
from .dataset import messages
from .plan import resolve


class MultiCADEnv:
    def __init__(self, engine, workspace, *, python=None):
        self.engine = engine
        self.root = Path(workspace).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Use a fresh session workspace")
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = MultiCADBackend(engine, python=python)
        self.task = None

    def reset(self, task, *, asset_root=None):
        if self.task is not None:
            raise RuntimeError("Create a new environment per episode")
        self.task = copy.deepcopy(task)
        self.actions = []
        self.history = []
        self.steps = 0
        self.done = False
        self.goals = []
        for relative in task.get("goal_images", []):
            if asset_root is None:
                raise ValueError("An asset root is required for goal images")
            p = (Path(asset_root).resolve() / relative).resolve()
            if not p.is_relative_to(Path(asset_root).resolve()) or not p.is_file():
                raise ValueError("Goal image escaped or is missing")
            self.goals.append(str(p))
        self.initial = copy.deepcopy(task["initial_actions"])
        _, objects = resolve(self.initial, allow_empty=True, return_objects=True)
        self.state = {"objects": objects, "geometry": None}
        self.current = []
        if self.initial:
            result = self.backend.run(self.initial, self.root / "initial")
            if not result["execution_ok"]:
                raise RuntimeError("Initial native state failed")
            self._update(result, self.root / "initial")
        else:
            empty = self.root / "empty.json"
            empty.write_text('{"vertices":[],"faces":[]}')
            self.current = render_mesh(
                empty, self.root / "empty-views", size=384, views=("iso",)
            )
        return self.observation()

    def _update(self, result, directory):
        _, objects = resolve(
            self.initial + self.actions, allow_empty=True, return_objects=True
        )
        self.state = {
            "objects": objects,
            "geometry": result.get(
                "geometry", result.get("observation", {}).get("geometry")
            ),
        }
        self.current = render_mesh(
            directory / "mesh.json", directory / "views", size=384, views=("iso",)
        )

    def observation(self):
        return {
            "state": copy.deepcopy(self.state),
            "goal_images": list(self.goals),
            "images": list(self.current),
            "remaining_steps": max(0, self.task["max_steps"] - self.steps),
            "engine": self.engine,
        }

    def messages(self):
        observation = self.observation()
        return messages(
            self.engine,
            self.task,
            observation["state"],
            self.history,
            [*self.goals, *self.current],
        )

    def step(self, action):
        if self.task is None or self.done:
            raise RuntimeError("Reset first; cannot step a terminated episode")
        self.steps += 1
        reward = 0.0
        info = {"execution_ok": False}
        try:
            action = validate_action(action)
            candidate = self.initial + self.actions + [action]
            features, objects = resolve(
                candidate, allow_empty=True, return_objects=True
            )
            if action["op"] == "finish" and not features:
                raise ValueError("Cannot finish an empty part")
            if features:
                directory = self.root / f"step-{self.steps:03d}"
                result = self.backend.run(
                    self.actions + [action],
                    directory,
                    initial_actions=self.initial,
                    reference_actions=self.task["reference_actions"]
                    if action["op"] == "finish"
                    else None,
                    expected_metrics=self.task["analytic_metrics"]
                    if action["op"] == "finish"
                    else None,
                )
                if not result["execution_ok"]:
                    raise ValueError(result.get("error", "Native execution failed"))
                self.actions.append(action)
                self._update(result, directory)
                reward = result["reward"]
                info = {
                    "execution_ok": True,
                    "validation": result.get("validation", {}),
                }
            else:
                self.actions.append(action)
                self.state = {"objects": objects, "geometry": None}
                info = {
                    "execution_ok": True,
                    "feedback_scope": "Validated feature graph; no solid exists yet",
                }
            self.done = action["op"] == "finish"
        except (ValueError, KeyError, IndexError) as exc:
            info["error"] = str(exc)
        self.done |= self.steps >= self.task["max_steps"]
        self.history.append(
            {
                "action": action,
                "response": json.dumps(action),
                "execution_ok": info["execution_ok"],
                **({"error": info["error"]} if "error" in info else {}),
            }
        )
        return self.observation(), reward, self.done, info
