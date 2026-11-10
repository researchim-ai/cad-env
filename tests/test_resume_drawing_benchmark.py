import pytest

from scripts.resume_drawing_benchmark import pending_cases


def test_resume_keeps_successes_and_model_failures_but_retries_infrastructure():
    selected = [{"task_id": "a"}, {"task_id": "b"}]
    rows = [
        {"task_id": "a", "standard": "iso", "status": "complete", "task_success": True},
        {
            "task_id": "a",
            "standard": "gost",
            "status": "complete",
            "task_success": False,
        },
        {"task_id": "b", "standard": "iso", "status": "infrastructure_error"},
    ]
    assert [(p["task_id"], s) for p, s in pending_cases(selected, rows)] == [
        ("b", "iso"),
        ("b", "gost"),
    ]


@pytest.mark.parametrize("bad", ["duplicate", "unexpected"])
def test_resume_rejects_corrupt_saved_episode_keys(bad):
    row = {"task_id": "a", "standard": "iso", "status": "complete"}
    rows = [row, row] if bad == "duplicate" else [{**row, "task_id": "other"}]
    with pytest.raises(ValueError):
        pending_cases([{"task_id": "a"}], rows)
