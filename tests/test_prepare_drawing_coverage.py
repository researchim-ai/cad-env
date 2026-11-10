import json

import pytest

from scripts.prepare_drawing_coverage import prepare


def fixture(tmp_path):
    pair = {
        "task_id": "part",
        "split_group": "parent",
        "split": "train",
        "family": "plate",
        "representations": {},
    }
    for standard in ("iso", "gost"):
        root = tmp_path / standard
        (root / "exports").mkdir(parents=True)
        task = root / "task.json"
        task.write_text(
            json.dumps(
                {
                    "split": "train",
                    "split_group": "parent",
                    "reference_actions": [{"op": "finish"}],
                }
            )
        )
        row = {
            "task_id": "part",
            "split_group": "parent",
            "step_id": 0,
            "completion": [{"content": [{"text": '{"op":"finish"}'}]}],
        }
        (root / "exports/train.feedback.jsonl").write_text(json.dumps(row) + "\n")
        pair["representations"][standard] = {"dataset": str(root), "task": str(task)}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    heldout = tmp_path / "heldout.json"
    heldout.write_text('[{"split_group":"different-parent"}]')
    return heldout


def test_paired_rows_keep_source_indices_and_hashes(tmp_path):
    heldout = fixture(tmp_path)
    plan = prepare(tmp_path, [heldout], 1, 42)
    assert plan["training_rows"] == 2
    assert plan["selected"][0]["representations"]["iso"]["row_index"] == 0
    assert len(plan["sources"]) == 2
    assert len(plan["heldout_sha256"][str(heldout)]) == 64


@pytest.mark.parametrize("tamper", ["heldout", "teacher", "task_lineage"])
def test_rejects_leakage_and_different_teachers(tmp_path, tamper):
    heldout = fixture(tmp_path)
    if tamper == "heldout":
        heldout.write_text('[{"split_group":"parent"}]')
    elif tamper == "teacher":
        path = tmp_path / "gost/exports/train.feedback.jsonl"
        path.write_text(path.read_text().replace("finish", "create_body"))
    else:
        path = tmp_path / "gost/task.json"
        path.write_text(path.read_text().replace('"train"', '"validation"'))
    with pytest.raises(ValueError):
        prepare(tmp_path, [heldout], 1, 42)
