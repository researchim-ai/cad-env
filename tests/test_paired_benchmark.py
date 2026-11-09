import json

from scripts.benchmark_paired_drawings import select_pairs, summarize


def test_selection_uses_only_validation_and_distinct_parents(tmp_path):
    rows = []
    for family in ("plate", "fixture", "flange"):
        for tier in ("challenge", "extreme"):
            for i in range(10):
                identity = f"{family}-{tier}-{i}"
                rows.append(
                    {
                        "task_id": identity,
                        "split_group": identity,
                        "family": family,
                        "difficulty": tier,
                        "split": "validation" if i < 8 else "train",
                    }
                )
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = select_pairs(tmp_path, 20)
    assert result == select_pairs(tmp_path, 20)
    assert len({r["split_group"] for r in result}) == 20
    assert all(r["split"] == "validation" for r in result)


def test_infrastructure_failure_is_not_counted_as_model_failure():
    rows = [
        {
            "standard": "iso",
            "status": "complete",
            "task_success": True,
            "reward": 1,
            "stop_reason": "agent_finish",
            "steps": 40,
            "invalid_actions": 0,
        },
        {"standard": "iso", "status": "infrastructure_error"},
    ]
    report = summarize(rows)["iso"]
    assert report["evaluated"] == 1 and report["infrastructure_errors"] == 1
    assert report["success_rate"] == report["mean_reward"] == 1


def test_analysis_refuses_an_incomplete_benchmark(tmp_path):
    import pytest

    from scripts.analyze_paired_benchmark import analyze

    (tmp_path / "status.json").write_text(json.dumps({"status": "evaluating"}))
    with pytest.raises(ValueError, match="complete"):
        analyze(tmp_path)
