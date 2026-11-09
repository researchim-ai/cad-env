import copy
from collections import defaultdict

from cad_env.learning.release_data import revisit_group, specs
from cad_env.learning.visual_tasks import make_visual_group


def test_shared_state_does_not_reveal_counterfactual_goal_and_teaches_finish():
    group = make_visual_group(8000002, selected_count=4, entity_labels=True)
    frozen = copy.deepcopy(group)
    revised = revisit_group(group, 8000002)
    assert group == frozen
    assert all(t["initial_actions"] == revised[0]["initial_actions"] for t in revised)
    assert all(t["instruction"] == revised[0]["instruction"] for t in revised)
    assert revised[0]["oracle_actions"] == [{"op": "finish"}]
    assert all(len(t["oracle_actions"]) > 1 for t in revised[1:])
    for before, after in zip(group, revised):
        assert before["reference_actions"] == after["reference_actions"]
        assert before["goal_spec"] == after["goal_spec"]


def test_fresh_generation_balances_edits_preserves_natural_splits_and_excludes_test():
    tasks = specs(5, 5)
    assert len(tasks) == 80
    parents = defaultdict(set)
    phases = defaultdict(set)
    for t in tasks:
        assert t["split"] in ("train", "validation")
        parents[t["split"]].add(t["split_group"])
        phases[t["condition_group"]].add(t["task_id"])
        assert t["goal_spec"]["entity_labels"]
        assert t["goal_spec"]["angle"] == 0
        assert len(t["oracle_actions"]) <= t["max_steps"]
    assert len(parents["train"]) == len(parents["validation"]) == 5
    assert not parents["train"] & parents["validation"]
    assert all(len(ids) == 4 for ids in phases.values())
    all_ids = {t["task_id"] for t in tasks}
    for t in tasks:
        assert t["counterfactual_goal_images"][0].split("/")[1] in all_ids
    denied = parents["train"] | parents["validation"]
    assert all(t["split_group"] not in denied for t in specs(5, 5, denied))
