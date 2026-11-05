from collections import defaultdict

from cad_env.learning.visual_units import build_tasks


def test_units_are_paired_without_leaking_value_holdout_or_physical_parents():
    tasks = build_tasks(12, 2)
    assert len(tasks) == 256
    groups = defaultdict(list)
    physical = defaultdict(set)
    train_values, ood_values = set(), set()
    for t in tasks:
        groups[t["condition_group"]].append(t)
        physical[t["split_group"]].add(t["split"])
        value = t["oracle_actions"][0]["value"]
        if t["evaluation_tier"] == "unit_ood":
            assert t["split"] == "test" and len(t["goal_spec"]["selected"]) <= 4
            ood_values.add(value)
        else:
            train_values.add(value)
        for path in t["counterfactual_goal_images"]:
            assert path.split("/")[1] in {x["task_id"] for x in tasks}
    assert train_values.isdisjoint(ood_values)
    assert all(len(v) == 1 for v in physical.values())
    assert all(len(v) == 4 for v in groups.values())
    for variants in groups.values():
        assert len({t["instruction"] for t in variants}) == 1
    board = tasks[:16]
    assert len({t["split_group"] for t in board}) == 1
    assert "inches" in board[4]["instruction"] and " mm" in board[0]["instruction"]
    assert board[0]["reference_actions"] == board[4]["reference_actions"]
