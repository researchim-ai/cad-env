"""Migrate public teacher history without rewriting frozen native datasets."""

import json

from cad_env.agent.observations import normalize_single_action_prompt


def normalize_feedback_prompt(messages):
    messages = normalize_single_action_prompt(messages)
    decoder = json.JSONDecoder()
    for message in messages:
        if message["role"] != "user" or not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") != "text":
                continue
            text = block["text"]
            try:
                payload, end = decoder.raw_decode(text)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            changed = False
            for transition in payload.get("recent_transitions", []):
                if "response" not in transition and isinstance(
                    transition.get("action"), dict
                ):
                    transition["response"] = json.dumps(transition["action"])[:2000]
                    changed = True
            if changed:
                block["text"] = json.dumps(payload) + text[end:]
    return messages
