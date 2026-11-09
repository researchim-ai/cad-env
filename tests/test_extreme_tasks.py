import copy
from collections import defaultdict

import pytest

from cad_env.agent.observations import agent_messages
from cad_env.agent.schema import MAX_ACTIONS, validate_program
from cad_env.learning.extreme_tasks import (
    FAMILIES,
    TIERS,
    make_extreme_task,
    render_extreme_goal,
    select_tasks,
)


def test_action_limit_128_and_reject_129():
    assert MAX_ACTIONS == 128
    assert (
        len(
            validate_program(
                [{"op": "finish"}] * 0
                + [{"op": "create_body", "name": "Body"}] * 127
                + [{"op": "finish"}]
            )
        )
        == 128
    )
    with pytest.raises(ValueError, match="128 actions"):
        validate_program(
            [{"op": "create_body", "name": "Body"}] * 128 + [{"op": "finish"}]
        )
    with pytest.raises(ValueError, match="final action"):
        validate_program([{"op": "finish"}, {"op": "create_body", "name": "Body"}])


@pytest.mark.parametrize("family", FAMILIES)
def test_extreme_tasks_are_feasible_programs_with_independent_hole_volumes(family):
    for seed in range(20):
        challenge = make_extreme_task(family, seed, "challenge")
        task = make_extreme_task(family, seed)
        assert len(task["oracle_actions"]) > 64
        assert len(task["oracle_actions"]) > len(challenge["oracle_actions"])
        assert len(task["oracle_actions"]) <= task["max_steps"] <= 128
        assert task["analytic_metrics"]["volume"] > 0
        assert not any(c.isdigit() for c in task["instruction"])
        cuts = task["parameters"]["cuts"]
        assert len({c["depth"] for c in cuts}) >= 2
        assert all(
            c["cb_depth"] < c["depth"] for c in cuts if c["cb_radius"] is not None
        )
        names = [a["name"] for a in task["oracle_actions"] if "name" in a]
        assert len(set(names)) == len(names)


def test_natural_splits_and_modality_pairs():
    tasks = select_tasks(1)
    assert len(tasks) == len(FAMILIES) * len(TIERS) * 2 * 2
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t["split_group"]].append(t)
        assert t["split"] in ("train", "validation")
        rebuilt = make_extreme_task(
            t["family"], t["generator_seed"], t["difficulty"], t["input_modality"]
        )
        assert rebuilt == t
    for rows in grouped.values():
        assert len(rows) == 2
        assert {r["input_modality"] for r in rows} == {"text", "image"}
        assert rows[0]["oracle_actions"] == rows[1]["oracle_actions"]
        assert rows[0]["split"] == rows[1]["split"]


def test_paginated_goal_contains_all_schedules_and_no_state_mutation(tmp_path):
    task = make_extreme_task("heterogeneous_plate", 3)
    before = copy.deepcopy(task)
    pages = render_extreme_goal(task, tmp_path / "goal.png")
    assert len(pages) == 3  # geometry + two schedules, 16 rows each
    assert all(p.is_file() for p in pages)
    assert task == before
    messages = agent_messages(
        task,
        {
            "state": {},
            "remaining_steps": task["max_steps"],
            "goal_images": list(map(str, pages)),
            "images": [],
        },
        visual=True,
    )
    text = messages[-1]["content"][0]["text"]
    assert "first 3 images are sheets of the TASK GOAL" in text
    assert "oracle_actions" not in text


def test_native_128_action_program(tmp_path):
    from cad_env.agent.backend import FreeCADBackend

    try:
        backend = FreeCADBackend(timeout=120)
    except RuntimeError:
        pytest.skip("Native FreeCAD required")
    actions = [
        {"op": "create_body", "name": "Body"},
        {
            "op": "sketch_rectangle",
            "name": "Base",
            "body": "Body",
            "length": 40,
            "width": 30,
        },
        {"op": "pad", "name": "Pad", "body": "Body", "profile": "Base", "length": 6},
    ]
    actions += [
        {"op": "set_parameter", "target": "Pad", "parameter": "length", "value": 6}
    ] * 124
    actions += [{"op": "finish"}]
    result = backend.run(
        actions,
        tmp_path / "native-128",
        reference_actions=actions,
        record=True,
        expected_metrics={"volume": 7200, "bounds": [0, 0, 0, 40, 30, 6]},
        reward_mode="strict_success",
    )
    assert result["execution_ok"], result.get("error")
    assert result["validation"]["task_success"]
    assert len(result["events"]) == 128
