import copy
import json

import pytest

from scripts.tune_paired_drawings import prepare_plan


def setup_sources(tmp_path):
    primary, paired, baseline = (
        tmp_path / name for name in ("primary", "paired", "baseline")
    )
    for p in (primary, paired, baseline):
        p.mkdir()
    roots = [primary / "iso", paired / "gost"]
    for root, standard in zip(roots, ("iso", "gost")):
        (root / "exports").mkdir(parents=True)
        (root / "episodes" / "train-task").mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps({"drawing_standard": standard}))
        (root / "episodes/train-task/task.json").write_text(
            json.dumps({"split": "train", "split_group": "train-parent"})
        )
        rows = [
            {
                "task_id": "train-task",
                "step_id": i,
                "completion": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"op": "pocket", "name": f"Hole{i}CB"}
                                ),
                            }
                        ],
                    }
                ],
            }
            for i in range(6)
        ]
        (root / "exports/train.feedback.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (primary / "batch-summary.json").write_text(
        json.dumps(
            {"status": "complete", "results": [{"name": "iso-000", "dataset": "iso"}]}
        )
    )
    (paired / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shards": [{"name": "iso-000", "added_dataset": str(roots[1])}],
            }
        )
    )
    (paired / "request.json").write_text(json.dumps({"primary": str(primary)}))
    (paired / "pairs.jsonl").write_text(
        json.dumps({"split": "validation", "split_group": "heldout-parent"}) + "\n"
    )
    (baseline / "status.json").write_text(json.dumps({"status": "complete"}))
    (baseline / "selection.json").write_text(
        json.dumps([{"split_group": "heldout-parent"}])
    )
    return paired, baseline, roots


def test_training_plan_has_unique_rows_and_matched_profile_targets(tmp_path):
    paired, baseline, _ = setup_sources(tmp_path)
    plan = prepare_plan(paired, baseline, rows=4)
    assert plan["training_rows"] == 4
    assert plan["standard_rows"] == {"iso": 2, "gost": 2}
    assert plan["unique_train_parents"] == 1
    assert plan["frozen_evaluation_groups_excluded"] == 1
    assert (
        plan["sources"][0]["selected_indices"] == plan["sources"][1]["selected_indices"]
    )


def test_training_plan_rejects_hidden_validation_lineage(tmp_path):
    paired, baseline, roots = setup_sources(tmp_path)
    task = roots[0] / "episodes/train-task/task.json"
    task.write_text(json.dumps({"split": "train", "split_group": "heldout-parent"}))
    with pytest.raises(ValueError, match="entered training"):
        prepare_plan(paired, baseline, rows=4)


def test_training_plan_rejects_mismatched_profile_teachers(tmp_path):
    paired, baseline, roots = setup_sources(tmp_path)
    target = roots[1] / "exports/train.feedback.jsonl"
    rows = [json.loads(s) for s in target.read_text().splitlines()]
    for row in rows:
        row["completion"] = copy.deepcopy(row["completion"])
        row["completion"][0]["content"][0]["text"] = '{"op":"finish"}'
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="different causal steps"):
        prepare_plan(paired, baseline, rows=4)
