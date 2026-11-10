import copy
import json

import pytest

from cad_env.freecad_v2.backend import Backend
from cad_env.freecad_v2.compositions import FAMILIES, LEVELS, make_task
from cad_env.freecad_v2.schema import validate


def test_all_levels_are_deterministic_bounded_and_specs_have_no_teacher_actions():
    for family in FAMILIES:
        counts = []
        for level in LEVELS:
            task = make_task(family, level, 10000)
            assert task == make_task(family, level, 10000)
            assert validate(task["actions"]) == task["actions"]
            assert "actions" not in json.dumps(task["design_spec"])
            assert "Cut" not in json.dumps(task["design_spec"])
            assert task["expected_volume"] > 0
            assert all(
                h["depth"] > h["cb_depth"] > 0 and h["cb_radius"] > h["radius"]
                for h in task["design_spec"]["holes"]
            )
            counts.append(len(task["actions"]))
        assert counts[0] < counts[1] < counts[2] <= 128


@pytest.mark.parametrize("family", FAMILIES)
def test_composed_native_parts_and_omitted_counterbore_are_distinguished(
    tmp_path, family
):
    task = make_task(family, "easy", 101)
    backend = Backend()
    correct = backend.run(
        task["actions"],
        tmp_path / "correct",
        reference=task["actions"],
        expected_volume=task["expected_volume"],
    )
    assert correct["execution_ok"], correct
    assert correct["validation"]["task_success"], correct
    missing = copy.deepcopy(task["actions"][:-3])
    missing.append({"op": "finish", "targets": [missing[-1]["name"]]})
    wrong = backend.run(missing, tmp_path / "omitted", reference=task["actions"])
    assert wrong["execution_ok"], wrong
    assert not wrong["validation"]["task_success"]
    assert wrong["validation"]["relative_symmetric_difference"] > 1e-5


def test_split_is_independent_of_profile_and_no_relabeling():
    for seed in range(20):
        t = make_task("gusset_bracket", "medium", seed)
        bucket = int(t["split_group"][:8], 16) % 10
        assert t["split"] == (
            "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        )


def test_dimensional_lines_have_unambiguous_coordinates_and_no_action_json():
    from cad_env.freecad_v2.drawings import specification_lines

    task = make_task("gusset_bracket", "hard", 4)
    for standard in ("iso", "gost"):
        lines = specification_lines(task["design_spec"], standard)
        assert all(len(line) <= 105 for line in lines)
        assert not any("{" in line or "[" in line for line in lines)
        for i in range(1, 25):
            assert any(line.startswith(f"H{i:02d} XYZ (") for line in lines)
        assert any("N (1; 0; 0)" in line for line in lines)


def test_balanced_plans_do_not_relabel_or_overlap():
    from collections import Counter

    from scripts.generate_v2_curriculum import plan

    groups = []
    for split in ("train", "validation", "test"):
        selected = plan(split, 1, 123)
        assert len(selected) == 9
        assert all(row["task"]["split"] == split for row in selected)
        assert Counter(row["task"]["difficulty"] for row in selected) == dict.fromkeys(
            LEVELS, 3
        )
        groups.append({row["task"]["split_group"] for row in selected})
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))


def test_inspect_is_paged_and_normal_state_is_compact(tmp_path):
    task = make_task("gusset_bracket", "hard", 4)
    backend = Backend()
    result = backend.run(
        task["actions"],
        tmp_path / "normal",
        reference=task["actions"],
        expected_volume=task["expected_volume"],
    )
    assert result["execution_ok"], result
    assert len(json.dumps(result["state"])) < 50000
    assert all("edges" not in item for item in result["state"])
    actions = task["actions"][:-1] + [
        {"op": "inspect", "target": "Base", "offset": 2, "limit": 2}
    ]
    inspected = backend.run(actions, tmp_path / "inspect")
    assert inspected["execution_ok"], inspected
    base = next(x for x in inspected["state"] if x["name"] == "Base")
    assert [x["index"] for x in base["edges"]] == [3, 4]
    assert base["edge_count"] > 2


def test_native_paired_drawing_hash_audit_and_crop_coverage(tmp_path):
    from cad_env.drawings.font_cache import require_font
    from cad_env.freecad_v2.drawings import audit_pair, export_pair, specification_lines

    try:
        require_font()
    except (ValueError, FileNotFoundError) as exc:
        pytest.skip(str(exc))
    task = make_task("flanged_bushing", "easy", 1)
    native = Backend().run(
        task["actions"],
        tmp_path / "native",
        reference=task["actions"],
        expected_volume=task["expected_volume"],
    )
    assert native["execution_ok"], native
    assert native["validation"]["task_success"]
    root = tmp_path / "paired"
    pair = export_pair(tmp_path / "native/final.step", task, root)
    assert audit_pair(root)["ok"]
    assert set(pair) == {"iso", "gost"}
    for standard in pair:
        manifest = json.loads((root / standard / "manifest.json").read_text())
        lines = specification_lines(task["design_spec"], standard)
        assert manifest["pages"] == 1 + (len(lines) + 29) // 30
        assert len(manifest["spec_crops"]) == sum(
            (min(30, len(lines) - start) + 5) // 6 for start in range(0, len(lines), 30)
        )
        assert all(c["min_font_px_at_budget"] >= 10.5 for c in manifest["spec_crops"])
        assert not manifest["can_claim_full_conformance"]
    path = root / "gost/design-spec.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        audit_pair(root)
