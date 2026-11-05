import copy

import pytest

from cad_env.learning.curriculum_experiment import validate_recipe


def test_unequal_budgets_and_duplicated_sources_cannot_be_queued():
    config = {
        "steps": 400,
        "arms": [
            {"name": n, "datasets": ["a", "b"], "budgets": [3, 5]}
            for n in ["control", "visual", "repair"]
        ],
    }
    validate_recipe(config)
    bad = copy.deepcopy(config)
    bad["arms"][1]["budgets"] = [3, 6]
    with pytest.raises(ValueError):
        validate_recipe(bad)
    bad = copy.deepcopy(config)
    bad["arms"][1]["datasets"] = ["a", "a"]
    with pytest.raises(ValueError):
        validate_recipe(bad)


def test_selection_rejects_a_high_scoring_model_that_forgets_old_tasks():
    from cad_env.learning.curriculum_experiment import select_arm

    reports = {
        n: {"metrics": {"success_at_1": v}}
        for n, v in [("replay", 0.3), ("visual", 0.7), ("repair", 0.8)]
    }
    evidence = {
        n: {"retention": {"success_at_1_delta": d}}
        for n, d in [("replay", 0), ("visual", -0.02), ("repair", -0.2)]
    }
    assert select_arm(reports, evidence, ["retention"]) == "visual"
    assert select_arm({"repair": reports["repair"]}, evidence, ["retention"]) is None
