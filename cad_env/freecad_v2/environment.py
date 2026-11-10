"""Single-action v2 agent loop with transactional replay and public-only observations."""

import copy
import json
from pathlib import Path

from .backend import Backend
from .schema import MAX_ACTIONS, TOOLS


def messages(instruction, observation, history, *, image_count=0):
    guide = {
        op: {"required": sorted(r), "optional": sorted(o)}
        for op, (r, o) in TOOLS.items()
    }
    result = [
        {
            "role": "system",
            "content": "Return one JSON action. Contract freecad.v2; mm, degrees. "
            "Sketch profiles: circle(radius), rectangle(length,width), polygon(points), "
            "segments(start,segments of line(to) or arc(mid,to)). Plane: origin, normal, x_axis. "
            "Pocket cuts against the sketch normal. Topology indices are 1-based and source-local; "
            "Use inspect(target, optional offset=0, limit=32) to retrieve a page of edges/faces before dress-ups; "
            "ordinary state gives object definitions and topology counts. Inspect again after topology edits. "
            "Patterns contain copies including the original; "
            "polar angle is total span (360 has no duplicated endpoint). finish requires solid targets. "
            + json.dumps(guide),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": instruction,
                    "observation": observation,
                    "history": history,
                },
                ensure_ascii=False,
            ),
        },
    ]
    for message in result:
        message["content"] = [{"type": "text", "text": message["content"]}]
    if image_count:
        result[-1]["content"][0]["text"] += (
            "\nThe attached images are views of the task GOAL; current geometry is described in JSON."
        )
        result[-1]["content"] += [{"type": "image"} for _ in range(image_count)]
    return result


class Environment:
    def __init__(self, output, backend=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.backend = backend or Backend()
        self.actions, self.history = [], []
        self.state, self.attempts, self.closed = [], 0, False

    def step(self, action, *, reference=None):
        if self.closed:
            raise ValueError("Episode closed")
        action = copy.deepcopy(action)
        self.attempts += 1
        folder = self.output / f"step-{self.attempts:03d}"
        try:
            result = self.backend.run(
                self.actions + [action], folder, reference=reference
            )
        except ValueError as exc:
            result = {
                "execution_ok": False,
                "error_type": "InvalidAction",
                "error": str(exc),
                "reward": 0.0,
            }
        if result["execution_ok"]:
            self.actions.append(copy.deepcopy(action))
            self.state = result["state"]
            self.closed = action["op"] == "finish"
        self.closed |= self.attempts >= MAX_ACTIONS
        observation = {
            "state": copy.deepcopy(self.state),
            "attempts": self.attempts,
            "remaining_actions": MAX_ACTIONS - self.attempts,
            "error": result.get("error"),
        }
        self.history.append(
            {
                "action": action,
                "execution_ok": result["execution_ok"],
                "error": result.get("error"),
            }
        )
        with (self.output / "trajectory.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {"action": action, "observation_after": observation, "info": result}
                )
                + "\n"
            )
        return observation, result.get("reward", 0), self.closed, result
