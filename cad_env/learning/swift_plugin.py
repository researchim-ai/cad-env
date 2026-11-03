"""EXPERIMENTAL ms-swift adapter for FreeCAD tool-action multi-turn GRPO.

Interface reference: KnowledgeXLab/IterCAD @ 55998c4a4537573f67ca241dedc52ee2840652ca.
CPU replay core is tested with real FreeCAD. This plugin is NOT a claim of a
validated ms-swift/vLLM distributed GPU run; that integration gate remains open.
No custom GRPO loss, advantage clamp, or policy-gradient monkey-patch is added.
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from cad_env.learning.multiturn import message_text, replay_conversation


def _replay(task_id, messages, visual):
    task_dir = os.environ["FREECAD_TASK_DIR"]
    output = Path(os.environ["FREECAD_ROLLOUT_DIR"]) / uuid.uuid4().hex
    return replay_conversation(task_dir, task_id, messages, output, visual=visual)


class FreeCADScheduler(MultiTurnScheduler):
    def check_finished(self, infer_request, response_choice, current_turn):
        try:
            action = json.loads(message_text(response_choice.message.content))
        except ValueError:
            action = {}
        return (
            isinstance(action, dict) and action.get("op") == "finish"
        ) or super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn):
        # The framework already appends the current assistant response before step.
        task_id = (infer_request.data_dict or {})["task_id"]
        result = _replay(task_id, infer_request.messages, visual=True)
        observation = result["observation"]
        paths = observation.get("images", [])
        feedback = json.dumps(
            {k: v for k, v in observation.items() if k != "images"}, ensure_ascii=False
        )
        feedback += "\nCurrent FreeCAD state. Continue with one JSON action; finish only when complete."
        if paths:
            infer_request.images.extend(paths)
            feedback += "\n" + "\n".join("<image>" for _ in paths)
        infer_request.messages.append({"role": "user", "content": feedback})
        return {
            "infer_request": infer_request,
            "rollout_infos": {
                "freecad_steps": result["steps"],
                "freecad_invalid_actions": result["invalid_actions"],
            },
        }


class FreeCADTerminalReward(ORM):
    def __call__(self, completions, messages, task_id, **kwargs):
        if not len(completions) == len(messages) == len(task_id):
            raise ValueError("Rollout metadata lengths differ")
        with ThreadPoolExecutor(
            max_workers=int(os.environ.get("FREECAD_REWARD_WORKERS", "2"))
        ) as pool:
            return list(
                pool.map(
                    lambda pair: _replay(pair[0], pair[1], False)["reward"],
                    zip(task_id, messages),
                )
            )


multi_turns["freecad_tools"] = FreeCADScheduler
orms["freecad_terminal"] = FreeCADTerminalReward
