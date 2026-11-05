import json

from cad_env.learning.grpo_audit import audit
from cad_env.learning.visual_bandit import VisualBanditReward


def test_saved_reward_audit_detects_false_positive(monkeypatch, tmp_path):
    action = {
        "op": "set_parameter",
        "target": "S",
        "parameter": "diameter_0",
        "value": 12,
    }
    task = {
        "task_id": "t",
        "split_group": "g",
        "goal_spec": {"selected": [0]},
        "initial_actions": [
            {"op": "sketch_circles", "name": "S", "circles": [{"radius": 3}]}
        ],
        "reference_actions": [
            {"op": "sketch_circles", "name": "S", "circles": [{"radius": 6}]}
        ],
    }
    monkeypatch.setattr(
        "cad_env.learning.grpo_audit.load_private_task", lambda *a, **k: task
    )
    (tmp_path / "cad-training-run.json").write_text(
        json.dumps({"arguments": {}, "gradient_signal": {}})
    )
    path = tmp_path / "native-rollouts/t/sample/score.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"task_id": "t", "completion": json.dumps(action), "reward": 1})
    )
    assert audit(tmp_path, tmp_path)["disagreements"] == []
    action["value"] = 10
    path.write_text(
        json.dumps({"task_id": "t", "completion": json.dumps(action), "reward": 1})
    )
    assert len(audit(tmp_path, tmp_path)["disagreements"]) == 1


def test_reward_ledger_keeps_zero_signal_batches(tmp_path, monkeypatch):
    reward = VisualBanditReward(tmp_path, tmp_path / "scores", workers=1)
    monkeypatch.setattr(
        reward,
        "score",
        lambda task_id, completion: {
            "task_id": task_id,
            "completion": completion,
            "reward": 0.0,
        },
    )
    assert reward(["a", "b"], ["t", "t"]) == [0, 0]
    ledger = json.loads((reward.output / "reward-batches.jsonl").read_text())
    assert ledger["batch_index"] == 0
    assert ledger["task_group_statistics"][0]["zero_variance"]
    assert [s["sample_index"] for s in ledger["samples"]] == [0, 1]
