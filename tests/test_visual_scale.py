from collections import defaultdict

from cad_env.learning.visual_scale import build_tasks


def test_fresh_curriculum_preserves_counterfactuals_and_lineage():
    tasks = build_tasks(16, 4)
    assert len(tasks) == 160
    groups = defaultdict(list)
    physical = defaultdict(set)
    for task in tasks:
        groups[task["condition_group"]].append(task)
        physical[task["split_group"]].add(task["split"])
        n = len(task["goal_spec"]["selected"])
        assert len(task["oracle_actions"]) == n + 1
        if n > 4:
            assert task["split"] == "test"
        assert task["oracle_actions"][-1] == {"op": "finish"}
    assert all(len(v) == 1 for v in physical.values())
    for variants in groups.values():
        assert len(variants) == 4
        assert len({str(t["goal_spec"]["selected"]) for t in variants}) == 4
        assert all(
            t["initial_actions"] == variants[0]["initial_actions"] for t in variants
        )
        assert all(t["instruction"] == variants[0]["instruction"] for t in variants)
