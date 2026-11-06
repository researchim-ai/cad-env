import json

import pytest

from cad_env.learning.protocol import digest
from cad_env.learning.visual_diagnostics import compare_slices


def test_visual_diagnostics_separate_units_and_labels_and_count_failures(tmp_path):
    summary = {"benchmark": "visual-grounding-v6-units"}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    rows = []
    for i in range(4):
        path = tmp_path / "episodes" / str(i)
        path.mkdir(parents=True)
        (path / "task.json").write_text(
            json.dumps(
                {
                    "instruction": "diameter 0.5 inches"
                    if i % 2
                    else "diameter 12.7 mm",
                    "goal_spec": {
                        "selected": [0, 1] if i > 1 else [0],
                        "entity_labels": i > 1,
                    },
                }
            )
        )
        rows.append(
            {
                "task_id": str(i),
                "split_group": str(i // 2),
                "attempt": 0,
                "task_success": False,
            }
        )
    before = {
        "status": "complete",
        "dataset_digest": digest(summary),
        "protocol_digest": "same",
        "task_ids": [str(i) for i in range(4)],
        "episodes": rows,
    }
    after = {
        **before,
        "episodes": [{**r, "task_success": r["task_id"] == "0"} for r in rows],
    }
    report = compare_slices(tmp_path, before, after)
    assert report["slices"]["labels:no"]["after_success"] == 0.5
    assert report["slices"]["units:mm"]["after_success"] == 0.5
    assert report["slices"]["units:inches"]["after_success"] == 0
    assert report["slices"]["edits:2"]["after_success"] == 0
    with pytest.raises(ValueError):
        compare_slices(tmp_path, before, {**after, "limited_evaluation": True})
