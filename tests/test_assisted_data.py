import copy
import pytest
from cad_env.learning.assisted_data import transform_task


def task(split="validation", selected=None):
    return {
        "split": split,
        "mode": "edit",
        "instruction": "Change marked holes.",
        "public_task": {"instruction": "Change marked holes."},
        "goal_spec": {"angle": 90, "entity_labels": False, "selected": selected or [0]},
        "oracle_actions": [{"op": "finish"}],
        "reference_actions": [{"op": "finish"}],
    }


def test_observation_variant_preserves_private_targets_and_does_not_mutate():
    original = task()
    frozen = copy.deepcopy(original)
    changed = transform_task(original, True)
    assert original == frozen
    for k in ["oracle_actions", "reference_actions", "split", "mode"]:
        assert changed[k] == original[k]
    assert changed["goal_spec"]["selected"] == original["goal_spec"]["selected"]
    assert changed["goal_spec"]["angle"] == 0
    assert changed["goal_spec"]["entity_labels"]
    assert changed["public_task"]["instruction"] == changed["instruction"]


def test_public_instruction_cannot_reveal_private_selected_subset():
    assert (
        transform_task(task(selected=[0]))["instruction"]
        == transform_task(task(selected=[1, 2]))["instruction"]
    )


def test_transform_never_reads_test_or_relabels_an_already_labelled_task():
    with pytest.raises(ValueError, match="TRAIN/validation"):
        transform_task(task("test"))
    t = task()
    t["goal_spec"]["entity_labels"] = True
    with pytest.raises(ValueError, match="already"):
        transform_task(t)
