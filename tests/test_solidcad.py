import copy
import os
import sys
from pathlib import Path

import pytest

from cad_env.solidcad.backend import SolidBackend
from cad_env.solidcad.schema import arc_points, validate
from cad_env.solidcad.tasks import FAMILIES, make_task


def test_solid_graph_contract_is_bounded():
    for program in [
        [{"op": "python", "code": "print(1)"}],
        [{"op": "finish", "targets": ["Missing"]}],
        [
            {"op": "box", "name": "B", "length": 1, "width": 2, "height": float("nan")},
            {"op": "finish", "targets": ["B"]},
        ],
    ]:
        with pytest.raises(ValueError):
            validate(program)
    assert len(arc_points([0, 0], [1, 1], [2, 0])) == 128
    with pytest.raises(ValueError):
        arc_points([0, 0], [1, 0], [2, 0])


@pytest.mark.parametrize("engine", ["freecad", "cadquery"])
@pytest.mark.parametrize("family", FAMILIES)
def test_native_advanced_feature_and_wrong_size(tmp_path, engine, family):
    if engine == "cadquery":
        pytest.importorskip("cadquery")
    try:
        backend = SolidBackend(
            engine, python=os.environ.get("MULTICAD_PYTHON", sys.executable)
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))
    task = make_task(family, 3)
    result = backend.run(
        task["program"],
        tmp_path / "correct",
        reference=task["program"],
        expected_volume=task["expected_volume"],
    )
    assert result["execution_ok"], result
    assert result["validation"]["task_success"], result
    assert result["geometry"]["solids"] == (2 if family == "two_bodies" else 1)
    if task["base_volume"] is not None:
        assert 0 < result["geometry"]["volume"] < task["base_volume"]
    assert (Path(result["output_dir"]) / result["artifact"]).is_file()
    wrong = copy.deepcopy(task["program"])
    if "height" in wrong[0]:
        wrong[0]["height"] += 1
    else:
        wrong[0]["points"][2][1] += 1
        wrong[0]["points"][3][1] += 1
    bad = backend.run(wrong, tmp_path / "wrong", reference=task["program"])
    assert bad["execution_ok"], bad
    assert not bad["validation"]["task_success"], bad
