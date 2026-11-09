import pytest

from cad_env.agent.schema import validate_program
from cad_env.learning.industrial_parts import (
    FAMILIES,
    LEVELS,
    PartBuilder,
    make_industrial_task,
)
from scripts.generate_industrial_drawings import select_plan


@pytest.mark.parametrize("family", FAMILIES)
def test_deterministic_valid_parts_and_increasing_structure(family):
    lengths = []
    for level in LEVELS:
        for seed in range(5):
            task = make_industrial_task(family, seed, level)
            assert task == make_industrial_task(family, seed, level)
            assert validate_program(task["oracle_actions"]) == task["oracle_actions"]
            names = [a["name"] for a in task["oracle_actions"] if "name" in a]
            assert len(names) == len(set(names))
            assert task["analytic_metrics"]["volume"] > 0
            assert len(task["oracle_actions"]) <= task["max_steps"] <= 128
            assert task["design_checks"]
            assert all(
                c["actual_mm"] >= c["minimum_mm"] - 1e-6 for c in task["design_checks"]
            )
        lengths.append(len(task["oracle_actions"]))
    assert lengths[0] < lengths[1] < lengths[2]


def test_bearing_seat_matches_sourced_boundary_dimensions():
    for level in LEVELS:
        t = make_industrial_task("bearing_cartridge", 1, level)
        h = next(h for h in t["parameters"]["cuts"] if h["name"] == "Shaft")
        assert h["radius"] == 5.5  # original nominal shaft clearance, not a sourced fit
        assert h["cb_radius"] == 15
        assert h["cb_depth"] == 9
        assert t["design_provenance"]["bearing_boundary_reference"]["outside_mm"] == 30
        assert not t["design_provenance"]["manufacturing_approval"]


def test_plans_disjoint_by_natural_geometry_split():
    def groups(plan):
        return {t["split_group"] for s in plan["shards"] for t in s["tasks"]}

    train = select_plan()
    val = select_plan(split="validation")
    assert len(groups(train)) == len(groups(val)) == 18
    assert not groups(train) & groups(val)
    for plan in [train, val]:
        for shard in plan["shards"]:
            for t in shard["tasks"]:
                bucket = int(t["split_group"][:8], 16) % 100
                assert (bucket < 75) if t["split"] == "train" else (75 <= bucket < 90)
    assert not groups(val) & groups(
        select_plan(split="validation", excluded=groups(val))
    )


def test_builder_rejects_intersection_and_invalid_counterbore():
    b = PartBuilder()
    b.rectangle(100, 80, 10)
    with pytest.raises(ValueError, match="clearance"):
        b.hole("Edge", "BasePad", 1, 1, 4, 10)
    b.hole("First", "BasePad", 30, 30, 3, 10)
    with pytest.raises(ValueError, match="intersect"):
        b.hole("Overlap", "BasePad", 32, 30, 3, 10)
    with pytest.raises(ValueError, match="counterbore"):
        b.hole("BadCB", "BasePad", 60, 50, 3, 10, 5, 12)
