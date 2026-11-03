"""One public prompt contract shared by teacher data and deployed policies."""

import copy
import json

from .schema import ACTION_GUIDE
from .tasks import public_task


def agent_messages(task, observation, history=(), *, visual=False):
    state = observation.get("state", observation)
    context = {
        "task": public_task(task),
        "state": state,
        "remaining_steps": observation.get("remaining_steps", task["max_steps"]),
        "recent_transitions": list(history)[-4:],
    }
    if observation.get("error"):
        context["last_error"] = observation["error"]
    system = "You control FreeCAD through typed CAD tools.\n" + ACTION_GUIDE
    system += "\nFor this sequential task, return exactly ONE JSON action object. Inspect current state/visual feedback, recover from errors, and finish only when complete."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    images = list(observation.get("images", [])) if visual else []
    if images:
        messages[-1]["content"] = [
            {
                "type": "text",
                "text": messages[-1]["content"]
                + "\nImages show the CURRENT candidate, not a hidden target.",
            }
        ] + [{"type": "image"} for _ in images]
        messages[0]["content"] = [{"type": "text", "text": system}]
    return messages


def transition(action, info, observation):
    return {
        "action": copy.deepcopy(action),
        "execution_ok": info.get("execution_ok", False),
        "error": observation.get("error"),
    }
