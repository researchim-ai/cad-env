"""Cross-engine semantic and native geometry regressions (optional runtimes)."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from cad_env.multicad.backend import MultiCADBackend
from cad_env.multicad.plan import resolve
from cad_env.multicad.tasks import FAMILIES, make_task


def test_shared_split_and_support_edit():
    for family in FAMILIES:
        tasks = [
            make_task(family, 4, mode, modality)
            for mode in ("create", "edit", "repair")
            for modality in ("text", "image")
        ]
        assert len({t["split_group"] for t in tasks}) == 1
        assert len({t["split"] for t in tasks}) == 1
        edit = tasks[2]
        assert resolve(edit["initial_actions"] + edit["oracle_actions"]) == resolve(
            edit["reference_actions"]
        )
    bad = copy.deepcopy(tasks[0]["reference_actions"])
    bad.insert(
        -1,
        {"op": "set_parameter", "target": "BasePad", "parameter": "width", "value": 10},
    )
    with pytest.raises(ValueError):
        resolve(bad)


def test_reject_bad_graph_and_no_arbitrary_code():
    for actions in [
        [{"op": "python", "code": "print(1)"}],
        [{"op": "create_body", "name": "B"}, {"op": "create_body", "name": "C"}],
    ]:
        with pytest.raises(ValueError):
            resolve(actions)
    assert resolve([{"op": "create_body", "name": "Body"}], allow_empty=True) == []


def test_removed_engine_is_not_accepted():
    from cad_env.solidcad.backend import SolidBackend

    for backend in (MultiCADBackend, SolidBackend):
        with pytest.raises(ValueError):
            backend("blender")


@pytest.mark.parametrize("engine", ["freecad", "cadquery"])
@pytest.mark.parametrize("family", FAMILIES)
def test_native_geometry_and_wrong_dimensions(tmp_path, engine, family):
    python = os.environ.get("MULTICAD_PYTHON", sys.executable)
    if engine == "cadquery":
        import importlib.util

        if not importlib.util.find_spec("cadquery") and python == sys.executable:
            pytest.skip("CadQuery runtime required")
    try:
        backend = MultiCADBackend(engine, python=python)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    task = make_task(family, 7)
    good = backend.run(
        task["oracle_actions"],
        tmp_path / "good",
        reference_actions=task["reference_actions"],
        expected_metrics=task["analytic_metrics"],
    )
    assert good["execution_ok"], good
    assert good["validation"]["task_success"], good
    artifact = Path(good["output_dir"]) / good["artifact"]
    assert artifact.is_file() and artifact.stat().st_size > 0
    wrong = copy.deepcopy(task["oracle_actions"])
    wrong[2]["length"] += 2
    bad = backend.run(
        wrong,
        tmp_path / "bad",
        reference_actions=task["reference_actions"],
        expected_metrics=task["analytic_metrics"],
    )
    assert bad["execution_ok"], bad
    assert not bad["validation"]["task_success"]
    unfinished = backend.run(
        task["oracle_actions"][:-1],
        tmp_path / "unfinished",
        reference_actions=task["reference_actions"],
    )
    assert not unfinished["validation"]["task_success"]


@pytest.mark.parametrize("engine", ["freecad", "cadquery"])
def test_native_sequential_edit_and_rejection(tmp_path, engine):
    from cad_env.multicad.environment import MultiCADEnv

    if engine == "cadquery":
        pytest.importorskip("cadquery")
    try:
        env = MultiCADEnv(engine, tmp_path / "session")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    task = make_task("stepped_bore", 0, "edit", "text")
    obs = env.reset(task)
    assert obs["state"]["geometry"]["volume"] > 0
    before = copy.deepcopy(env.actions)
    obs, reward, done, info = env.step(
        {"op": "set_parameter", "target": "Missing", "parameter": "length", "value": 5}
    )
    assert not info["execution_ok"] and env.actions == before and not done
    for action in task["oracle_actions"]:
        obs, reward, done, info = env.step(action)
    assert done and reward == 1 and info["validation"]["task_success"]
    with pytest.raises(RuntimeError):
        env.step({"op": "finish"})


def test_native_dataset_audit_and_training_gate(tmp_path):
    from cad_env.learning.train import check_training_source
    from cad_env.multicad.dataset import audit, generate

    try:
        MultiCADBackend("freecad")
    except RuntimeError:
        pytest.skip("FreeCAD required")
    root = tmp_path / "corpus"
    summary = generate(
        root,
        engines=("freecad",),
        seeds=1,
        families=("stepped_bore",),
        modes=("create", "edit"),
        modalities=("image",),
    )
    assert summary["accepted"] == 2 and not summary["rejected"]
    assert audit(root)["ok"]
    source = root / "exports/train.feedback.jsonl"
    assert check_training_source(root, source)["ok"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    task = json.loads(
        (root / "episodes" / rows[0]["task_id"] / "task.json").read_text()
    )
    assert rows[0]["images"][:2] == task["goal_images"]
    assert "reference_actions" not in json.dumps(rows[0]["prompt"])
    from cad_env.multicad.environment import MultiCADEnv

    env = MultiCADEnv("freecad", tmp_path / "parity-session")
    env.reset(task, asset_root=root)
    assert env.messages() == rows[0]["prompt"]
    env.step(task["oracle_actions"][0])
    assert env.messages() == rows[1]["prompt"]
    rows[0]["completion"][0]["content"] = '{"op":"finish"}'
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert not audit(root)["ok"]
    with pytest.raises(ValueError):
        check_training_source(root, source)


def test_geometric_group_ignores_hole_order_and_names():
    from cad_env.multicad.tasks import geometry_group

    program = make_task("blind_pocket_grid", 0)["reference_actions"]
    groups = [program[i : i + 2] for i in range(3, len(program) - 1, 2)]
    reordered = (
        program[:3]
        + [action for pair in reversed(groups) for action in pair]
        + [program[-1]]
    )
    assert reordered != program and geometry_group(reordered) == geometry_group(program)
    mapping = {a["name"]: "Other" + a["name"] for a in program if "name" in a}
    renamed = [
        {
            k: mapping.get(v, v) if k in {"name", "body", "profile", "support"} else v
            for k, v in a.items()
        }
        for a in program
    ]
    assert geometry_group(renamed) == geometry_group(program)
