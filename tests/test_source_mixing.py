import pytest
from datasets import Dataset

from cad_env.learning.mixing import mix_sources
from cad_env.multicad.tasks import make_task


def test_balanced_source_budget_is_deterministic():
    a = Dataset.from_dict({"source": ["a"] * 2, "index": [0, 1]})
    b = Dataset.from_dict({"source": ["b"] * 9, "index": list(range(9))})
    d = mix_sources([a, b], 6)
    assert len(d) == 12 and d["source"].count("a") == d["source"].count("b") == 6
    assert d.to_dict() == mix_sources([a, b], 6).to_dict()
    assert len(set(d.select(range(6, 12))["index"])) == 6
    with pytest.raises(ValueError):
        mix_sources([a], 0)


def test_varied_edits_keep_targets_and_splits_without_fixed_delta():
    deltas = set()
    for seed in range(30):
        old = make_task("stepped_bore", seed, "edit")
        new = make_task("stepped_bore", seed, "edit", varied_edits=True)
        assert (
            old["reference_actions"] == new["reference_actions"]
            and old["split_group"] == new["split_group"]
        )
        deltas.add(
            new["initial_actions"][2]["length"] - new["oracle_actions"][0]["value"]
        )
    assert len(deltas) >= 4


def test_per_source_budgets_keep_exact_weights_and_reject_invalid_lengths():
    a = Dataset.from_dict({"x": ["a"] * 2})
    b = Dataset.from_dict({"x": ["b"] * 7})
    result = mix_sources([a, b], [3, 5])
    assert result["x"].count("a") == 3 and result["x"].count("b") == 5
    for budgets in [[3], [3, 0], [3, True], [3, 1.5]]:
        with pytest.raises(ValueError):
            mix_sources([a, b], budgets)


def test_explicit_seeds_keep_common_anchors_identical_when_other_sources_added():
    clean = Dataset.from_dict({"id": list(range(20))})
    anchor = Dataset.from_dict({"id": list(range(100, 120))})
    extra = Dataset.from_dict({"id": list(range(200, 220))})
    control = mix_sources([clean, anchor], [8, 5], source_seeds=[42, 43])
    candidate = mix_sources(
        [clean, extra, anchor], [4, 6, 5], source_seeds=[42, 99, 43]
    )
    assert list(control["id"])[:4] == list(candidate["id"])[:4]
    assert list(control["id"])[-5:] == list(candidate["id"])[-5:]
    for seeds in [[1], [True, 2], [1, "2"]]:
        with pytest.raises(ValueError):
            mix_sources([clean, anchor], [2, 2], source_seeds=seeds)
