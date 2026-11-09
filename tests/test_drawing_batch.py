import json
from collections import Counter

import pytest

from cad_env.drawings.batch import run_batch, select_plan, verify_tasks


def test_500_plan_has_independent_balanced_geometries_and_natural_splits():
    plan = select_plan()
    assert plan == select_plan()
    rows = [(s["standard"], t) for s in plan["shards"] for t in s["tasks"]]
    assert len(rows) == len({t["split_group"] for _, t in rows}) == 500
    assert len({t["task_id"] for _, t in rows}) == 500
    assert Counter(s for s, _ in rows) == {"iso": 250, "gost": 250}
    for standard in ("iso", "gost"):
        tasks = [t for s, t in rows if s == standard]
        assert Counter(t["split"] for t in tasks) == {"train": 200, "validation": 50}
        assert Counter(t["difficulty"] for t in tasks) == {
            "challenge": 125,
            "extreme": 125,
        }
        assert all(
            t["input_modality"] == "image" and t["max_steps"] <= 128 for t in tasks
        )
        assert "Geometry schedule:" not in "".join(t["instruction"] for t in tasks)
    excluded = {rows[0][1]["split_group"]}
    assert not excluded.intersection(
        t["split_group"]
        for s in select_plan(excluded_groups=excluded)["shards"]
        for t in s["tasks"]
    )


def test_invalid_plan_and_mismatched_resume_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="multiple"):
        select_plan(501)
    (tmp_path / "plan.json").write_text(json.dumps(select_plan(10)))
    with pytest.raises(ValueError, match="differs"):
        run_batch(tmp_path, count=20)


def test_source_plan_guard_rejects_missing_tasks(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("")
    task = select_plan(10)["shards"][0]["tasks"][0]
    with pytest.raises(ValueError, match="accepted"):
        verify_tasks(tmp_path, [task])
