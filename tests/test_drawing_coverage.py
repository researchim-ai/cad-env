import pytest

from cad_env.learning.drawing_coverage import select_decisions


def rows():
    return [
        {
            "split": "train",
            "split_group": f"p{i % 10}",
            "step_id": i,
            "operation": op,
            "counterbore": cb,
            "progress": progress,
            "recovery": False,
        }
        for i, (op, cb, progress) in enumerate(
            [("sketch_circles", True, 0.2)] * 30
            + [("pocket", True, 0.3)] * 30
            + [("finish", False, 1)] * 20
            + [("pocket", False, 0.8)] * 30
            + [("pad", False, 0.1)] * 40
        )
    ]


def test_quotas_unique_parents_and_reproducibility():
    selected, report = select_decisions(rows(), 100)
    assert (selected, report) == select_decisions(rows(), 100)
    assert len({(r["split_group"], r["step_id"]) for r in selected}) == 100
    assert report["unique_parents"] == 10
    for cat, minimum in [
        ("cb_sketch", 20),
        ("cb_pocket", 20),
        ("finish", 10),
        ("late", 20),
    ]:
        assert report["categories"][cat] >= minimum
    assert report["recovery_available"] == 0


def test_shortage_is_reported_without_duplicating():
    data = rows()[-40:]
    selected, report = select_decisions(data, 30)
    assert len(selected) == 30
    assert report["quota_shortages"]["cb_sketch"] == 6


@pytest.mark.parametrize("kind", ["validation", "duplicate", "budget"])
def test_rejects_unsafe_or_impossible_pool(kind):
    data = rows()
    if kind == "validation":
        data[0]["split"] = "validation"
    elif kind == "duplicate":
        data.append(data[0])
    with pytest.raises(ValueError):
        select_decisions(data, 1000 if kind == "budget" else 10)
