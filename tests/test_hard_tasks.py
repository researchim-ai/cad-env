from collections import Counter, defaultdict

from cad_env.learning.hard_tasks import construction_tasks, make_hard_task, visual_tasks


def test_hard_construction_has_natural_disjoint_modality_pairs():
    tasks = construction_tasks()
    groups = defaultdict(list)
    for t in tasks:
        groups[t["split_group"]].append(t)
        assert t["reward_mode"] == "strict_success"
        assert len(t["oracle_actions"]) <= t["max_steps"]
        if t["input_modality"] == "image":
            assert not any(c.isdigit() for c in t["instruction"])
    assert len(tasks) == 304
    assert Counter(rows[0]["split"] for rows in groups.values()) == {
        "train": 96,
        "validation": 40,
        "test": 16,
    }
    for group, rows in groups.items():
        assert len(rows) == 2
        assert rows[0]["reference_actions"] == rows[1]["reference_actions"]
        bucket = int(group[:8], 16) % 100
        assert rows[0]["split"] == (
            "train" if bucket < 75 else "validation" if bucket < 90 else "test"
        )
    denied = next(iter(groups))
    assert all(t["split_group"] != denied for t in construction_tasks({denied}))


def test_counterbores_and_hollow_shafts_require_composed_features():
    plate = make_hard_task("counterbored_plate", 4000000)
    pockets = [a for a in plate["oracle_actions"] if a["op"] == "pocket"]
    assert len(pockets) == 3
    assert len(plate["parameters"]["holes"]) == 7
    shaft = make_hard_task("hollow_stacked_shaft", 4000001)
    assert len(shaft["parameters"]["stages"]) in {5, 6}
    assert len(shaft["oracle_actions"]) in {16, 18}
    assert [a["name"] for a in shaft["oracle_actions"] if a["op"] == "pocket"] == [
        "AxialBore",
        "MountHoles",
    ]


def test_hard_visual_never_labels_entities_and_keeps_counterfactuals_inside_split():
    tasks = visual_tasks()
    by_id = {t["task_id"]: t for t in tasks}
    assert len(tasks) == 736
    assert {
        len(t["goal_spec"]["selected"]) for t in tasks if t["split"] == "train"
    } == {2, 3, 4, 5, 6}
    assert any(t["goal_spec"]["angle"] % 90 for t in tasks)
    for t in tasks:
        assert not t["goal_spec"].get("entity_labels")
        assert len(t["oracle_actions"]) <= t["max_steps"]
        for image in t["counterfactual_goal_images"]:
            other = by_id[image.split("/")[1]]
            assert other["split"] == t["split"]
            assert other["split_group"] == t["split_group"]
            assert other["initial_actions"] == t["initial_actions"]
            assert other["goal_spec"]["selected"] != t["goal_spec"]["selected"]
