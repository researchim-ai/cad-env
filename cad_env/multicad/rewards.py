"""TRAIN-only native scoring hook for whole-program multi-engine RL rollouts."""

import json
import uuid
from pathlib import Path

from cad_env.agent.schema import validate_program

from .backend import MultiCADBackend
from .dataset import audit


class MultiCADReward:
    def __init__(self, dataset, output, *, python=None):
        self.root = Path(dataset).resolve()
        self.out = Path(output).resolve()
        report = audit(self.root)
        if not report["ok"]:
            raise ValueError("Multi-engine dataset audit failed")
        self.tasks = {
            r["task_id"]: r
            for line in (self.root / "manifest.jsonl").read_text().splitlines()
            if (r := json.loads(line))["accepted"] and r["split"] == "train"
        }
        self.python = python

    def score(self, task_id, completion):
        if task_id not in self.tasks:
            raise ValueError("Only registered TRAIN tasks may produce training rewards")
        row = self.tasks[task_id]
        task = json.loads((self.root / "episodes" / task_id / "task.json").read_text())
        try:
            actions = validate_program(json.loads(completion))
        except (ValueError, TypeError):
            return 0.0
        backend = MultiCADBackend(row["engine"], python=self.python)
        try:
            result = backend.run(
                actions,
                self.out / task_id / uuid.uuid4().hex,
                initial_actions=task["initial_actions"],
                reference_actions=task["reference_actions"],
                expected_metrics=task["analytic_metrics"],
                mesh=False,
            )
        except (ValueError, KeyError, IndexError):
            return 0.0
        return float(result.get("reward", 0.0))

    def __call__(self, completions, task_id, **kwargs):
        if len(completions) != len(task_id):
            raise ValueError("Every completion needs its registered identity")

        def text(completion):
            if isinstance(completion, str):
                return completion
            if isinstance(completion, list) and len(completion) == 1:
                content = completion[0].get("content", "")
                return (
                    content
                    if isinstance(content, str)
                    else "".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                )
            return ""

        return [
            self.score(identity, text(completion))
            for identity, completion in zip(task_id, completions)
        ]
