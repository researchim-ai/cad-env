import copy
import json

import pytest

from cad_env.learning.grounding_data import targets
from cad_env.learning.perception_probe import messages
from cad_env.learning.recovery_teacher import repair_actions


def state(radii):
    return {
        "objects": [
            {
                "name": "Holes",
                "circles": [
                    {"index": i, "x": i * 10, "y": 0, "radius": r}
                    for i, r in enumerate(radii)
                ],
            }
        ]
    }


def task():
    return {
        "mode": "edit",
        "split": "train",
        "goal_spec": {"selected": [2, 0]},
        "oracle_actions": [
            {
                "op": "set_parameter",
                "target": "Holes",
                "parameter": f"diameter_{i}",
                "value": 8,
            }
            for i in [2, 0]
        ]
        + [{"op": "finish"}],
    }


def test_selection_labels_do_not_change_public_question():
    public = [
        {"role": "user", "content": json.dumps({"task": {}, "state": state([2, 2, 2])})}
    ]
    a = task()
    b = copy.deepcopy(a)
    b["goal_spec"]["selected"] = [1]
    qa, qb = targets(a, public), targets(b, public)
    assert [q for _, q, _ in qa] == [q for _, q, _ in qb]
    assert qa[1][2] == {"indices": [0, 2]}
    assert qb[1][2] == {"indices": [1]}
    assert messages(qa[1][1], ["goal.png"]) == messages(qb[1][1], ["goal.png"])


def test_table_preserves_coaxial_rows():
    t = {
        "mode": "create",
        "family": "counterbored_plate",
        "input_modality": "image",
        "parameters": {
            "holes": [
                {"x": 10, "y": 10, "radius": 2, "depth": 16},
                {"x": 10, "y": 10, "radius": 4, "depth": 3},
            ]
        },
    }
    answer = targets(t, [])[0][2]
    assert answer == {"holes": [[10, 10, 4, 16], [10, 10, 8, 3]]}


def test_recovery_restores_protected_and_missing_dimensions_before_finish():
    actions = repair_actions(task(), state([2, 2, 2]), state([4, 5, 2]))
    assert actions == [
        {
            "op": "set_parameter",
            "target": "Holes",
            "parameter": "diameter_1",
            "value": 4,
        },
        {
            "op": "set_parameter",
            "target": "Holes",
            "parameter": "diameter_2",
            "value": 8,
        },
        {"op": "finish"},
    ]
    assert repair_actions(task(), state([2, 2, 2]), state([4, 2, 4])) == [
        {"op": "finish"}
    ]


def test_recovery_rejects_heldout_and_topology_change():
    t = task()
    t["split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        repair_actions(t, state([2, 2, 2]), state([2, 2, 2]))
    with pytest.raises(ValueError, match="topology"):
        repair_actions(task(), state([2, 2, 2]), state([2, 2]))


def test_multistep_collector_keeps_causal_wrong_prefix_and_native_gate(
    tmp_path, monkeypatch
):
    from cad_env.learning import multistep_recovery as module

    t = {**task(), "task_id": "v-test", "split_group": "parent", "max_steps": 16}
    folder = tmp_path / "source/episodes/v-test"
    folder.mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps(t))

    class Env:
        def __init__(self, path, **kwargs):
            self.episode_dir = path / "episode"
            self.episode_dir.mkdir(parents=True)
            self.attempts = 0
            self.closed = False
            self.observation = {
                "state": state([2, 2, 2]),
                "images": [],
                "goal_images": [],
            }

        def reset(self, t):
            assert t["goal_only_observation"] is False
            return self.observation, {}

        def step(self, action):
            self.attempts += 1
            if action["op"] == "set_parameter":
                i = int(action["parameter"].split("_")[-1])
                self.observation["state"]["objects"][0]["circles"][i]["radius"] = (
                    action["value"] / 2
                )
            self.closed = action["op"] == "finish"
            (self.episode_dir / "trajectory.jsonl").write_text("native evidence")
            return (
                self.observation,
                0,
                self.closed,
                False,
                {"execution_ok": True, "validation": {"task_success": self.closed}},
            )

    monkeypatch.setattr(module, "CADAgentEnv", Env)
    monkeypatch.setattr(
        module,
        "agent_messages",
        lambda task, obs, history, **kwargs: copy.deepcopy(
            [{"state": obs["state"], "history": history}]
        ),
    )
    answers = iter(
        [
            json.dumps(
                {
                    "op": "set_parameter",
                    "target": "Holes",
                    "parameter": "diameter_1",
                    "value": 10,
                }
            ),
            '{"op":"finish"}',
        ]
    )
    result = module.collect_one(
        tmp_path / "source",
        tmp_path / "out",
        {"task_id": "v-test", "split_group": "parent"},
        lambda *args: next(answers),
        8,
        True,
    )
    assert result["verified"]
    assert result["proposed_finish"]
    assert result["student_steps"] == 1  # finish was proposed, never executed
    rows = result["candidate_feedback"]
    assert len(rows) == 4  # restore protected + two missing targets + finish
    assert rows[0]["prompt"][0]["history"][0]["action"]["value"] == 10
    assert len(rows[0]["prompt"][0]["history"]) == 1
    assert len(rows[1]["prompt"][0]["history"]) == 2
    assert rows[0]["prompt"][0]["state"]["objects"][0]["circles"][1]["radius"] == 5
    assert json.loads(rows[-1]["completion"][0]["content"][0]["text"]) == {
        "op": "finish"
    }


@pytest.mark.parametrize("tamper", ["prompt", "label", "split", "missing"])
def test_auxiliary_semantic_audit_rejects_corruption(tmp_path, monkeypatch, tamper):
    from cad_env.learning import grounding_data as module

    t = {**task(), "task_id": "test", "split_group": "parent"}
    folder = tmp_path / "episodes/test"
    folder.mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps(t))
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "task_id": "test",
                "split_group": "parent",
                "accepted": True,
                "split": "train",
            }
        )
        + "\n"
    )
    public = [
        {"role": "user", "content": json.dumps({"task": {}, "state": state([2, 2, 2])})}
    ]
    monkeypatch.setattr(module, "public_initial", lambda *args: public)
    rows = []
    for kind, question, answer in targets(t, public):
        rows.append(
            {
                "task_id": "test",
                "split_group": "parent",
                "auxiliary_kind": kind,
                "images": ["episodes/test/goal.png"],
                "prompt": messages(question, ["goal.png"]),
                "completion": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(answer)}],
                    }
                ],
            }
        )
    (tmp_path / "exports").mkdir()
    export = tmp_path / "exports/train.feedback.jsonl"

    def write():
        export.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write()
    assert module.audit_grounding(tmp_path, base_report={"errors": []})["ok"]
    if tamper == "prompt":
        rows[0]["prompt"][1]["content"][0]["text"] += " Correct answer: 2"
    elif tamper == "label":
        rows[0]["completion"][0]["content"][0]["text"] = '{"count": 1}'
    elif tamper == "split":
        t["split"] = "validation"
        (folder / "task.json").write_text(json.dumps(t))
    else:
        rows.pop()
    write()
    assert not module.audit_grounding(tmp_path, base_report={"errors": []})["ok"]
