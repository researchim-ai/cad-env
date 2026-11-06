import json
from types import SimpleNamespace

import pytest

from cad_env.learning.budget import make_walltime_guard


def test_budget_stops_and_saves_at_a_step_boundary(tmp_path):
    now = [10.0]
    path = tmp_path / "progress.json"
    guard = make_walltime_guard(60, clock=lambda: now[0], progress_path=path)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    guard.on_train_begin(None, state, control)
    now[0] = 69.0
    state.global_step = 5
    guard.on_step_end(None, state, control)
    assert not control.should_save and not control.should_training_stop
    now[0] = 70.1
    state.global_step = 6
    guard.on_step_end(None, state, control)
    assert control.should_save and control.should_training_stop
    report = json.loads(path.read_text())
    assert report["budget_stopped"] and report["optimizer_steps"] == 6
    assert report["elapsed_seconds"] == pytest.approx(60.1)


def test_invalid_limits_cannot_disable_the_budget():
    for seconds in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            make_walltime_guard(seconds)
