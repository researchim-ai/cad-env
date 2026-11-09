import copy
import json

import pytest

from cad_env.agent.observations import transition
from cad_env.learning import recovery_export as module
from cad_env.learning.factory import teacher_transition
from cad_env.learning.perception_data import sha


def state(r):
    return {
        "objects": [
            {
                "name": "H",
                "circles": [
                    {"index": 0, "x": 0, "y": 0, "radius": 4},
                    {"index": 1, "x": 10, "y": 0, "radius": r},
                ],
            }
        ]
    }


def test_already_correct_source_tasks_can_be_repaired_after_student_damage():
    from cad_env.learning.recovery_teacher import repair_actions

    t = {"split": "train", "mode": "edit", "oracle_actions": [{"op": "finish"}]}
    assert repair_actions(t, state(2), state(2)) == [{"op": "finish"}]
    assert repair_actions(t, state(2), state(5))[0] == {
        "op": "set_parameter",
        "target": "H",
        "parameter": "diameter_1",
        "value": 4,
    }


@pytest.mark.parametrize(
    "tamper", [None, "future_prompt", "wrong_completion", "terminal_failure"]
)
def test_export_reconstructs_causal_native_continuation(tmp_path, monkeypatch, tamper):
    def public(task, obs, history, **kwargs):
        return copy.deepcopy([{"state": obs["state"], "history": history}])

    monkeypatch.setattr(module, "agent_messages", public)
    t = {
        "split": "train",
        "task_id": "test",
        "split_group": "parent",
        "mode": "edit",
        "oracle_actions": [{"op": "finish"}],
    }
    wrong = {
        "op": "set_parameter",
        "target": "H",
        "parameter": "diameter_1",
        "value": 10,
    }
    restore = {**wrong, "value": 4}
    finish = {"op": "finish"}
    clean = {"state": state(2), "images": [], "goal_images": ["goal.png"]}
    damaged = {"state": state(5), "images": [], "goal_images": ["goal.png"]}
    info = {"execution_ok": True, "result_dir": str(tmp_path)}
    events = [
        {
            "observation_before": clean,
            "observation_after": damaged,
            "action": wrong,
            "info": info,
        },
        {
            "observation_before": damaged,
            "observation_after": clean,
            "action": restore,
            "info": info,
        },
        {
            "observation_before": clean,
            "observation_after": clean,
            "action": finish,
            "info": {**info, "validation": {"task_success": True}},
        },
    ]
    history = [{**transition(wrong, info, damaged), "response": json.dumps(wrong)}]
    rows = []
    for i, action in enumerate([restore, finish]):
        obs = damaged if i == 0 else clean
        rows.append(
            {
                "task_id": "test",
                "split_group": "parent",
                "step_id": i + 1,
                "prompt": public(t, obs, history),
                "completion": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(action)}],
                    }
                ],
                "images": ["goal.png"],
            }
        )
        history.append(teacher_transition(action, info, clean))
    if tamper == "terminal_failure":
        events[-1]["info"]["validation"]["task_success"] = False
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text("".join(json.dumps(e) + "\n" for e in events))
    record = {
        "verified": True,
        "trajectory": str(trajectory),
        "trajectory_sha256": sha(trajectory),
        "student_steps": 1,
        "student_evidence": [
            {
                "prompt": public(t, clean, []),
                "images": ["goal.png"],
                "observation": clean,
                "response": json.dumps(wrong),
            }
        ],
        "candidate_feedback": copy.deepcopy(rows),
    }
    if tamper == "future_prompt":
        record["candidate_feedback"][0]["prompt"][0]["future_answer"] = restore
    if tamper == "wrong_completion":
        record["candidate_feedback"][0]["completion"][0]["content"][0]["text"] = (
            '{"op":"finish"}'
        )
    if tamper:
        with pytest.raises(ValueError):
            module.verified_rows(t, record)
    else:
        actual, _ = module.verified_rows(t, record)
        assert actual == rows
        assert len(actual[0]["prompt"][0]["history"]) == 1
        assert len(actual[1]["prompt"][0]["history"]) == 2
