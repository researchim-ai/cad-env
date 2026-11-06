import copy

from cad_env.learning.spatial_probe import variants


def test_paired_views_preserve_goal_and_geometry_without_mutating_task():
    task = {
        "goal_spec": {
            "angle": 120,
            "entity_labels": False,
            "selected": [1, 3],
            "circles": [{"x": 2, "y": 5, "radius": 1}],
        },
        "initial_actions": [{"op": "create_body", "name": "Body"}],
    }
    before = copy.deepcopy(task)
    views = variants(task)
    assert task == before
    assert set(views) == {
        "original_plain",
        "original_ids",
        "canonical_plain",
        "canonical_ids",
    }
    for name, view in views.items():
        assert view["goal_spec"]["selected"] == before["goal_spec"]["selected"]
        assert view["goal_spec"]["circles"] == before["goal_spec"]["circles"]
        assert view["initial_actions"] == before["initial_actions"]
        assert view["goal_spec"]["angle"] == (
            0 if name.startswith("canonical") else 120
        )
        assert view["goal_spec"]["entity_labels"] == name.endswith("_ids")


def test_entity_label_intervention_does_not_depend_on_selected_subset():
    task = {"goal_spec": {"angle": 30, "selected": [0], "entity_labels": False}}
    other = copy.deepcopy(task)
    other["goal_spec"]["selected"] = [1, 2]
    a, b = variants(task), variants(other)
    for condition in a:
        for field in ("angle", "entity_labels"):
            assert a[condition]["goal_spec"][field] == b[condition]["goal_spec"][field]
