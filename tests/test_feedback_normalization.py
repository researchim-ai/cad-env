import json

from cad_env.learning.feedback_normalization import normalize_feedback_prompt


def test_history_migration_is_public_idempotent_and_preserves_images_and_suffix():
    action = {"op": "create_body", "name": "B"}
    state = {
        "recent_transitions": [{"action": action, "execution_ok": True, "error": None}]
    }
    source = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(state) + "\nGOAL FIRST"},
                {"type": "image"},
            ],
        }
    ]
    output = normalize_feedback_prompt(source)
    assert "response" not in source[0]["content"][0]["text"]
    payload, end = json.JSONDecoder().raw_decode(output[0]["content"][0]["text"])
    assert payload["recent_transitions"][0]["response"] == json.dumps(action)
    assert output[0]["content"][0]["text"][end:] == "\nGOAL FIRST"
    assert output[0]["content"][1] == {"type": "image"}
    assert normalize_feedback_prompt(output) == output
