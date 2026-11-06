import json

import pytest

from cad_env.learning.protocol import digest
from cad_env.learning.visual_error_audit import analyze


def test_wrong_entity_and_wrong_unit_are_different_failures(tmp_path):
    root, output = tmp_path / "data", tmp_path / "evaluation"
    root.mkdir()
    output.mkdir()
    summary = {"benchmark": "test"}
    (root / "summary.json").write_text(json.dumps(summary))
    episodes = []
    for i in range(2):
        task = {
            "instruction": "Resize to diameter 0.5 inches"
            if i
            else "Resize to diameter 12.7 mm",
            "oracle_actions": [
                {
                    "op": "set_parameter",
                    "target": "S",
                    "parameter": "diameter_0",
                    "value": 12.7,
                },
                {"op": "finish"},
            ],
        }
        folder = root / "episodes" / str(i)
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps(task))
        action = {
            "op": "set_parameter",
            "target": "S",
            "parameter": "diameter_0" if i else "diameter_1",
            "value": 0.5 if i else 12.7,
        }
        trajectory = output / f"{i}.jsonl"
        trajectory.write_text(
            "\n".join(
                json.dumps({"action": a, "info": {"execution_ok": True}})
                for a in [action, {"op": "finish"}]
            )
        )
        episodes.append(
            {
                "task_id": str(i),
                "attempt": 0,
                "trajectory": trajectory.name,
                "task_success": False,
            }
        )
    report = {
        "status": "complete",
        "dataset_digest": digest(summary),
        "task_ids": ["0", "1"],
        "episodes": episodes,
    }
    (output / "evaluation.json").write_text(json.dumps(report))
    result = analyze(root, output)
    assert (
        result["first_action_entity_rate"] == 0.5
        and result["first_action_value_rate"] == 0.5
    )
    assert result["counts"]["inch_literal_used_as_mm"] == 1
    assert result["counts"]["episodes_with_unrequested_edits"] == 1
    assert result["counts"]["finish_before_requested_values"] == 2
    (output / "evaluation.json").write_text(
        json.dumps({**report, "limited_evaluation": True})
    )
    with pytest.raises(ValueError, match="unlimited"):
        analyze(root, output)
