import copy
import math
from pathlib import Path

import pytest

from cad_env.freecad_v2.backend import Backend
from cad_env.freecad_v2.environment import Environment
from cad_env.freecad_v2.schema import validate
from cad_env.freecad_v2.tasks import FAMILIES, make_task


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [3, 17, 99])
def test_native_operation_and_wrong_geometry(tmp_path, family, seed):
    task = make_task(family, seed)
    backend = Backend()
    correct = backend.run(
        task["actions"],
        tmp_path / "correct",
        reference=task["actions"],
        expected_volume=task["expected_volume"],
    )
    assert correct["execution_ok"], correct
    assert correct["validation"]["task_success"], correct
    if family in ("fillet", "chamfer"):
        base = correct["state"][1]["operation"]
        assert base["op"] == "pad"
        reference = task["actions"][:2] + [{"op": "finish", "targets": ["Base"]}]
        unmodified = backend.run(reference, tmp_path / "base")
        assert 0 < correct["geometry"]["volume"] < unmodified["geometry"]["volume"]
    wrong = copy.deepcopy(task["actions"][:-1])
    target = task["actions"][-1]["targets"][0]
    wrong += [
        {
            "op": "translate",
            "name": "WrongPosition",
            "source": target,
            "vector": [1, 0, 0],
        },
        {"op": "finish", "targets": ["WrongPosition"]},
    ]
    bad = backend.run(wrong, tmp_path / "wrong", reference=task["actions"])
    assert bad["execution_ok"], bad
    assert not bad["validation"]["task_success"]


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "python", "code": "bad"},
        {"op": ["pad"]},
        {"op": "sketch", "name": "P", "profile": {"type": []}},
        {
            "op": "sketch",
            "name": "P",
            "profile": {"type": "circle", "radius": float("nan")},
        },
        {
            "op": "sketch",
            "name": "P",
            "profile": {"type": "circle", "radius": 1},
            "normal": [0, 0, 0],
        },
        {"op": "loft", "name": "L", "profiles": ["Missing", "Missing"]},
        {"op": "finish", "targets": ["Missing"]},
    ],
)
def test_rejects_unsafe_actions(bad):
    with pytest.raises(ValueError):
        validate([bad], terminal=False)


def test_transactional_steps_recover_after_name_collision_and_no_reference_leak(
    tmp_path,
):
    task = make_task("parameter_edit", 8)
    env = Environment(tmp_path / "episode")
    obs, _, _, info = env.step(task["actions"][0], reference=task["actions"])
    assert info["execution_ok"], info
    obs, reward, done, info = env.step(task["actions"][0], reference=task["actions"])
    assert not info["execution_ok"] and not done and reward == 0
    assert len(env.actions) == 1
    assert set(obs) == {"state", "attempts", "remaining_actions", "error"}
    for action in task["actions"][1:]:
        obs, reward, done, info = env.step(action, reference=task["actions"])
        assert info["execution_ok"], info
        if action["op"] != "finish":
            assert reward == 0
    assert done and reward == 1


def test_bad_topology_is_structured_native_error(tmp_path):
    actions = make_task("fillet", 1)["actions"]
    next(a for a in actions if a["op"] == "fillet")["edges"] = [999]
    result = Backend().run(actions, tmp_path / "invalid")
    assert not result["execution_ok"]
    assert "index" in result["error"]


def test_bent_sweep_independent_volume_not_just_self_comparison(tmp_path):
    actions = [
        {"op": "sketch", "name": "P", "profile": {"type": "circle", "radius": 2}},
        {
            "op": "sweep",
            "name": "S",
            "profile": "P",
            "path": [[0, 0, 0], [0, 0, 20], [20, 0, 20]],
        },
        {"op": "finish", "targets": ["S"]},
    ]
    result = Backend().run(
        actions, tmp_path / "bent", reference=actions, expected_volume=math.pi * 4 * 40
    )
    assert result["execution_ok"], result
    assert result["validation"]["task_success"], result


def test_overlapping_patterns_do_not_double_count_volume(tmp_path):
    actions = make_task("linear_pattern", 1)["actions"]
    actions[2]["vector"] = [1, 0, 0]
    result = Backend().run(actions, tmp_path / "overlap")
    assert not result["execution_ok"]
    assert "Overlapping" in result["error"]


def test_generator_exports_audited_causal_training_rows(tmp_path):
    import json

    from cad_env.freecad_v2.__main__ import generate
    from cad_env.freecad_v2.environment import messages
    from cad_env.learning.train import check_training_source

    root = tmp_path / "data"
    summary = generate(root, seed=11, families=["tilted_pad", "parameter_edit"])
    assert summary["accepted"] == summary["attempted"] == 2
    assert check_training_source(root, root / "exports/train.feedback.jsonl")["ok"]
    for split in ["train", "validation", "test"]:
        for line in (root / f"exports/{split}.feedback.jsonl").read_text().splitlines():
            row = json.loads(line)
            task = json.loads(
                (root / "episodes" / row["task_id"] / "task.json").read_text()
            )
            trace = [
                json.loads(s)
                for s in (
                    root / "episodes" / row["task_id"] / "rollout/trajectory.jsonl"
                )
                .read_text()
                .splitlines()
            ]
            step = row["step_id"]
            observation = (
                trace[step - 1]["observation_after"]
                if step
                else {
                    "state": [],
                    "attempts": 0,
                    "remaining_actions": 128,
                    "error": None,
                }
            )
            history = [
                {
                    "action": e["action"],
                    "execution_ok": e["info"]["execution_ok"],
                    "error": e["info"].get("error"),
                }
                for e in trace[:step]
            ]
            assert row["prompt"] == messages(
                task["instruction"], observation, history, image_count=3
            )
            assert (
                json.loads(row["completion"][0]["content"][0]["text"])
                == task["actions"][step]
            )


def test_benchmark_loop_with_scripted_teacher_is_only_a_plumbing_test(
    tmp_path, monkeypatch
):
    import json
    import sys

    from cad_env.freecad_v2.__main__ import generate
    from scripts import benchmark_freecad_v2 as runner

    seed = next(
        s for s in range(100) if make_task("pocket", s)["split"] == "validation"
    )
    task = make_task("pocket", seed)
    root = tmp_path / "data"
    generate(root, seed=seed, families=["pocket"])

    class ScriptedTeacher:
        def __init__(self, *args, **kwargs):
            self.identity = {"test_double": True}
            self.actions = iter(task["actions"])

        def __call__(self, prompt, images):
            assert len(images) == 3
            assert all(Path(p).is_file() for p in images)
            assert "reference_actions" not in json.dumps(prompt)
            return json.dumps(next(self.actions))

    monkeypatch.setattr(runner, "HFPolicy", ScriptedTeacher)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    output = tmp_path / "benchmark"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--dataset",
            str(root),
            "--model",
            "scripted-test-double",
            "--output",
            str(output),
        ],
    )
    runner.main()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["model_identity"]["test_double"]
    assert status["episodes"][0]["task_success"]
