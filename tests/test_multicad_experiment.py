import copy

import pytest

from cad_env.learning.multicad_comparison import combined, compare, image_effect
from cad_env.learning.multicad_control import NoGoalPolicy


def report(engine, success, identity="same"):
    rows = [
        {
            "task_id": f"{engine}-{i}-{mode}",
            "split_group": str(i),
            "mode": mode,
            "task_success": success,
            "engine": engine,
        }
        for i in range(24)
        for mode in ["create", "edit"]
    ]
    return {
        "status": "complete",
        "dataset_sha256": engine + "-data",
        "protocol": {"max_steps": "task budget", "temperature": 0},
        "engine": engine,
        "split": "validation",
        "episodes": rows,
        "model_identity": identity,
        "metrics": {"success_at_1": float(success)},
    }


def test_clustered_comparison_refuses_changed_harness_and_incomplete_cases():
    before = report("blender", False)
    after = report("blender", True)
    r = compare(before, after)
    assert (
        r["independent_lineage_groups"] == 24
        and r["success_at_1_delta"] == 1
        and r["improvement_supported"]
    )
    bad = copy.deepcopy(after)
    bad["episodes"].pop()
    with pytest.raises(ValueError):
        compare(before, bad)
    bad = copy.deepcopy(after)
    bad["protocol"]["temperature"] = 0.8
    with pytest.raises(ValueError):
        compare(before, bad)
    all_engines = combined(
        [
            (report(e, False), report(e, True))
            for e in ["freecad", "cadquery", "blender"]
        ]
    )
    assert (
        all_engines["paired_task_count"] == 144
        and all_engines["independent_lineage_groups"] == 24
    )
    control = copy.deepcopy(before)
    control["image_condition"] = "no_goal"
    effects = image_effect(after, control)
    assert effects["modes"]["create"]["independent_lineage_groups"] == 24
    control["model_identity"] = "different"
    with pytest.raises(ValueError):
        image_effect(after, control)


def test_no_goal_control_preserves_text_current_image_and_original_messages(tmp_path):
    paths = []
    for name in ["spec", "goal", "current"]:
        p = tmp_path / name
        p.write_bytes(name.encode())
        paths.append(str(p))

    class Policy:
        temperature = 0.0
        identity = "model"

        def __call__(self, messages, image_paths):
            assert image_paths == paths[2:]
            assert messages[0]["content"][0]["text"] == "same text"
            assert sum(b["type"] == "image" for b in messages[0]["content"]) == 1
            return '{"op":"finish"}'

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "same text"}]
            + [{"type": "image"} for _ in paths],
        }
    ]
    original = copy.deepcopy(messages)
    control = NoGoalPolicy(Policy())
    assert control(messages, paths) == '{"op":"finish"}'
    assert messages == original and control.identity == "model"
    with pytest.raises(ValueError):
        control(messages, paths[:2])
