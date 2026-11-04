import copy

import pytest

from cad_env.learning.control_metrics import paired_image_control


def reports():
    full = {
        "status": "complete",
        "limited_evaluation": False,
        "dataset_digest": "data",
        "model_identity": {"files": {"weights": "fixed"}},
        "task_ids": [str(i) for i in range(100)],
        "protocol": {"attempts": 1, "max_steps": 8, "temperature": 0},
        "episodes": [
            {"task_id": str(i), "split_group": str(i // 4), "task_success": True}
            for i in range(100)
        ],
    }
    control = copy.deepcopy(full)
    control["protocol"]["image_condition"] = "blank_goal"
    for row in control["episodes"]:
        row["task_success"] = False
    return full, control


def test_image_control_clusters_counterfactual_siblings():
    full, control = reports()
    result = paired_image_control(full, control, resamples=100)
    assert result["matched_minus_control"] == 1
    assert result["cluster_bootstrap_95_ci"] == [1, 1]
    assert result["parent_groups"] == 25 and result["image_effect_supported"]


@pytest.mark.parametrize("change", ["model", "budget", "partial", "missing"])
def test_image_claim_rejects_unmatched_or_incomplete_evidence(change):
    full, control = reports()
    if change == "model":
        control["model_identity"] = {"files": {"weights": "different"}}
    elif change == "budget":
        control["protocol"]["max_steps"] = 80
    elif change == "partial":
        control["status"] = "running"
    else:
        control["episodes"].pop()
    with pytest.raises(ValueError):
        paired_image_control(full, control, resamples=100)
