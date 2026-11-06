import copy

import pytest

from cad_env.learning.visual_pilot import validate_recipe


def test_pilot_cannot_silently_become_a_long_or_multi_epoch_run():
    config = {
        "steps": 160,
        "train_minutes": 35,
        "total_minutes": 120,
        "save_steps": 40,
        "sources": [{"dataset": "a", "rows": 1280}, {"dataset": "b", "rows": 1280}],
    }
    validate_recipe(config)
    for key, value in [
        ("steps", 201),
        ("train_minutes", 46),
        ("total_minutes", 121),
        ("save_steps", 0),
    ]:
        bad = copy.deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_recipe(bad)
    bad = copy.deepcopy(config)
    bad["sources"][0]["rows"] = 10
    bad["sources"][1]["rows"] = 10
    with pytest.raises(ValueError, match="epoch"):
        validate_recipe(bad)
