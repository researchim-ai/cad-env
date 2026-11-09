"""One public prompt contract shared by teacher data and deployed policies."""

import copy
import json

from .schema import ACTION_GUIDE
from .tasks import public_task


def policy_images(observation):
    return list(observation.get("goal_images", [])) + list(
        observation.get("images", [])
    )


def normalize_single_action_prompt(messages):
    """Migrate the old contradictory whole-program line in frozen SFT rows."""
    messages = copy.deepcopy(messages)
    old_line = ACTION_GUIDE.splitlines()[0]
    new_line = 'Return ONLY one JSON action object with an "op" field.'
    for message in messages:
        if message["role"] != "system":
            continue
        content = message["content"]
        if isinstance(content, str):
            message["content"] = content.replace(old_line, new_line)
        else:
            for block in content:
                if block.get("type") == "text":
                    block["text"] = block["text"].replace(old_line, new_line)
    return messages


def agent_messages(task, observation, history=(), *, visual=False):
    state = observation.get("state", observation)
    if task.get("observation_profile") == "compact_v1":
        state = {
            **state,
            "objects": [
                {
                    k: v
                    for k, v in row.items()
                    if k
                    in (
                        "name",
                        "type",
                        "constraints",
                        "circles",
                        "length",
                        "dependencies",
                    )
                }
                for row in state.get("objects", [])
                if row["type"].startswith(("PartDesign::", "Sketcher::"))
            ],
        }
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
    goals = list(observation.get("goal_images", [])) if visual else []
    images = goals + (list(observation.get("images", [])) if visual else [])
    if images:
        messages[-1]["content"] = [
            {
                "type": "text",
                "text": messages[-1]["content"]
                + (
                    (
                        (
                            f"\nThe first {len(goals)} images are sheets of the TASK GOAL drawing. "
                            if len(goals) > 1
                            else "\nFIRST image is the TASK GOAL drawing/marker. "
                        )
                        + (
                            "Following image is the CURRENT candidate. "
                            if observation.get("images")
                            else "The CURRENT state is described in JSON. "
                        )
                        + "Read the goal image to determine the required action."
                    )
                    if goals
                    else "\nImages show the CURRENT candidate, not a hidden target."
                ),
            }
        ] + [{"type": "image"} for _ in images]
        messages[0]["content"] = [{"type": "text", "text": system}]
    return normalize_single_action_prompt(messages)


def transition(action, info, observation):
    return {
        "action": copy.deepcopy(action),
        "execution_ok": info.get("execution_ok", False),
        "error": observation.get("error"),
    }
