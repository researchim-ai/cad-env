"""Meaningful execution, geometry, supervision, and leakage checks."""

import copy
import json

import pytest

from cad_env.agent import FreeCADBackend, make_task, validate_action
from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.tasks import prompt_messages
from cad_env.learning.factory import generate_dataset, negative_actions
from cad_env.learning.rewards import (
    CADReward,
    group_statistics,
    parse_completion,
    score_completion,
)


@pytest.fixture(scope="module")
def backend():
    try:
        return FreeCADBackend(timeout=20)
    except RuntimeError:
        pytest.skip("Real FreeCAD console is required")


@pytest.mark.parametrize(
    "action",
    [
        {"op": "python", "code": "raise Exception()"},
        {"op": "create_body", "name": "../Body"},
        {"op": "create_body", "name": "Body", "output_path": "/tmp/x"},
        {
            "op": "pad",
            "name": "Pad",
            "body": "Body",
            "profile": "Sketch",
            "length": float("nan"),
        },
        {
            "op": "pad",
            "name": "Pad",
            "body": "Body",
            "profile": "Sketch",
            "length": True,
        },
        {
            "op": "set_parameter",
            "target": "Sketch",
            "parameter": "__class__",
            "value": 1,
        },
    ],
)
def test_reject_unbounded_or_arbitrary_actions(action):
    with pytest.raises(ValueError):
        validate_action(action)


def test_completion_contract():
    good = [{"op": "finish"}]
    assert parse_completion("```json\n" + json.dumps(good) + "\n```") == good
    for bad in [
        "[]",
        '{"actions": [], "reward": 1}',
        '[{"op":"finish"},{"op":"finish"}]',
        "not JSON",
    ]:
        with pytest.raises(ValueError):
            parse_completion(bad)


@pytest.mark.parametrize("family", ["plate", "bushing", "flange", "stepped_shaft"])
@pytest.mark.parametrize("mode", ["create", "edit", "repair"])
def test_real_oracles_and_wrong_dimensions(backend, tmp_path, family, mode):
    task = make_task(family, 7, mode)
    result = backend.run(
        task["oracle_actions"],
        tmp_path / "good",
        initial_actions=task["initial_actions"],
        reference_actions=task["reference_actions"],
        record=True,
    )
    assert result["execution_ok"], result.get("error")
    assert result["validation"]["task_success"], result.get("validation")
    assert result["validation"]["parametric"]
    assert result["validation"]["reward"] == 1.0
    assert (tmp_path / "good" / "output.FCStd").stat().st_size > 1000
    assert len(result["events"]) == len(task["oracle_actions"])
    wrong = score_completion(
        task, json.dumps(negative_actions(task)), tmp_path / "bad", backend
    )
    assert not wrong["task_success"]
    assert wrong["reward"] < 1.0


def test_invalid_native_operation_is_not_success(backend, tmp_path):
    task = make_task()
    actions = copy.deepcopy(task["oracle_actions"])
    actions[2]["profile"] = "MissingSketch"
    result = backend.run(actions, tmp_path)
    assert not result["execution_ok"]
    assert "MissingSketch" in result["error"]


def test_empty_and_noop_do_not_get_positive_labels(backend, tmp_path):
    task = make_task("bushing", 2, "edit")
    result = score_completion(task, '[{"op":"finish"}]', tmp_path, backend)
    assert not result["task_success"]
    assert result["reward"] < 1
    empty = score_completion(
        make_task(), '[{"op":"finish"}]', tmp_path / "empty", backend
    )
    assert empty["reward"] == 0


def test_sequential_recovery_and_termination(backend, tmp_path):
    task = make_task("plate", 3, "edit")
    env = CADAgentEnv(tmp_path, backend)
    obs, _ = env.reset(task)
    before = obs["state"]
    obs, reward, done, truncated, info = env.step(
        {"op": "set_parameter", "target": "Missing", "parameter": "length", "value": 3}
    )
    assert not info["execution_ok"] and not done and not truncated and reward == 0
    assert obs["state"] == before
    env.step(task["oracle_actions"][0])
    _, reward, done, truncated, info = env.step({"op": "finish"})
    assert done and reward == 1 and not truncated
    with pytest.raises(RuntimeError):
        env.step({"op": "finish"})
    assert len((env.episode_dir / "trajectory.jsonl").read_text().splitlines()) == 3


def test_step_budget(backend, tmp_path):
    task = make_task()
    task["max_steps"] = 1
    env = CADAgentEnv(tmp_path, backend)
    env.reset(task)
    _, _, done, truncated, _ = env.step({"op": "unknown"})
    assert truncated and not done


def test_split_lineage_and_private_reference():
    for seed in range(20):
        tasks = [
            make_task("plate", seed, mode) for mode in ("create", "edit", "repair")
        ]
        assert len({t["split_group"] for t in tasks}) == 1
        assert len({t["split"] for t in tasks}) == 1
        for task in tasks:
            serialized = json.dumps(prompt_messages(task))
            assert (
                "reference_actions" not in serialized
                and "oracle_actions" not in serialized
            )
    assert make_task("flange", 3) == make_task("flange", 3)


def test_factory_exports_only_verified_examples(backend, tmp_path):
    root = tmp_path / "data"
    summary = generate_dataset(
        root,
        seeds=1,
        families=("plate", "bushing"),
        modes=("create", "repair"),
        render=False,
        preferences=True,
        holdout_families=("bushing",),
    )
    assert summary["accepted"] == 4
    assert summary["rejected"] == 0
    all_rows = []
    for path in (root / "exports").glob("*.grpo.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["split"] = path.name.split(".")[0]
            all_rows.append(row)
            assert "completion" not in row and "oracle_actions" not in json.dumps(row)
            if row["family"] == "bushing":
                assert row["split"] == "test"
    groups = {}
    for row in all_rows:
        groups.setdefault(row["split_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    reward = CADReward(root / "episodes", tmp_path / "rollouts")
    task = make_task()
    values = reward(
        completions=[
            [{"role": "assistant", "content": json.dumps(task["oracle_actions"])}],
            "bad",
        ],
        task_id=[task["task_id"], task["task_id"]],
    )
    assert values == [1.0, 0.0]
    with pytest.raises(FileExistsError):
        generate_dataset(root, seeds=1, render=False)


def test_group_diagnostics():
    stats = group_statistics([1, 0.5, 0])
    assert abs(sum(stats["normalized_advantages"])) < 1e-8
    assert not stats["zero_variance"]
    assert group_statistics([0, 0])["zero_variance"]
    assert group_statistics([1, 1])["normalized_advantages"] == [0, 0]


def test_analytic_spec_catches_shared_reference_error(backend, tmp_path):
    from cad_env.agent.tasks import expected_metrics

    task = make_task("plate", 5)
    wrong_reference = negative_actions(task)
    result = backend.run(
        wrong_reference,
        tmp_path,
        reference_actions=wrong_reference,
        expected_metrics=expected_metrics(task),
    )
    assert result["execution_ok"]
    assert result["validation"]["geometry_score"] == 1.0
    assert not result["validation"]["spec_pass"]
    assert not result["validation"]["task_success"]


def test_actual_worker_timeout(tmp_path):
    task = make_task()
    result = FreeCADBackend(timeout=0.001).run(task["oracle_actions"], tmp_path)
    assert not result["execution_ok"] and result["error_type"] == "Timeout"
    assert not result.get("validation", {}).get("task_success", False)


def test_multiturn_group_records_repair_and_success(backend, tmp_path):
    from cad_env.learning.rollouts import collect_group

    task = make_task("plate", 5, "edit")
    position = [0]

    def policy(messages):
        i = position[0] % 3
        position[0] += 1
        if i == 0:
            return "not JSON"
        return json.dumps(task["oracle_actions"][i - 1])

    report = collect_group(task, policy, tmp_path, num_generations=2, backend=backend)
    assert report["successful_rollouts"] == 2
    assert report["zero_variance"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "successful.steps.sft.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert "not JSON" in json.dumps(rows[0]["prompt"])


def test_http_agent_session_isolation(backend, tmp_path):
    import asyncio

    import httpx

    from cad_env.agent.api import create_app

    async def run():
        app = create_app(tmp_path, executable=backend.executable)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/sessions", json={"family": "plate", "seed": 3, "mode": "edit"}
            )
            assert response.status_code == 200
            sid = response.json()["session_id"]
            assert (
                "reference_actions" not in response.text
                and "oracle_actions" not in response.text
            )
            task = make_task("plate", 3, "edit")
            for action in task["oracle_actions"]:
                result = await client.post(
                    f"/sessions/{sid}/step", json={"action": action}
                )
                assert result.status_code == 200
            assert result.json()["reward"] == 1.0
            assert result.json()["terminated"]
            assert "result_dir" not in result.text
            assert (
                await client.post(
                    f"/sessions/{sid}/step", json={"action": {"op": "finish"}}
                )
            ).status_code == 409
            assert (await client.delete(f"/sessions/{sid}")).status_code == 200
            assert (
                await client.post(
                    f"/sessions/{sid}/step", json={"action": {"op": "finish"}}
                )
            ).status_code == 404

    asyncio.run(run())


def test_render_depth_buffer_keeps_through_hole_open(backend, tmp_path):
    from PIL import Image

    from cad_env.agent.rendering import render_mesh

    task = make_task("plate", 3)
    result = backend.run(task["oracle_actions"], tmp_path / "cad", mesh=True)
    assert result["execution_ok"]
    paths = render_mesh(tmp_path / "cad" / "mesh.json", tmp_path / "views", size=128)
    image = Image.open(paths[1]).convert("RGB")
    assert image.getpixel((64, 64)) == (255, 255, 255)
    assert image.getpixel((48, 64)) != (255, 255, 255)
    assert (tmp_path / "views" / "view-cameras.json").is_file()


def test_dataset_audit_detects_tampering(backend, tmp_path):
    from cad_env.learning.audit import audit_dataset

    root = tmp_path / "data"
    generate_dataset(
        root, seeds=1, families=("plate",), modes=("create",), render=False
    )
    assert audit_dataset(root)["ok"]
    fcstd = next(root.glob("episodes/*/oracle/output.FCStd"))
    fcstd.write_bytes(b"not a FreeCAD file")
    report = audit_dataset(root)
    assert not report["ok"]
    assert any(
        "Hash mismatch" in error or "FCStd" in error for error in report["errors"]
    )


def test_vlm_export_loader_handles_typed_content(tmp_path):
    pytest.importorskip("datasets")
    from PIL import Image

    from cad_env.learning.train import load_export

    root = tmp_path / "data"
    (root / "exports").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "image.png")
    row = {
        "prompt": [
            {"role": "system", "content": "Build CAD"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Match the image"},
                    {"type": "image"},
                ],
            },
        ],
        "completion": [{"role": "assistant", "content": '[{"op":"finish"}]'}],
        "images": ["image.png"],
    }
    path = root / "exports" / "train.vlm.jsonl"
    path.write_text(json.dumps(row) + "\n")
    sample = load_export(path)[0]
    assert len(sample["images"]) == 1 and sample["images"][0].size == (16, 16)
    assert isinstance(sample["prompt"][0]["content"], list)
    assert isinstance(sample["completion"][0]["content"], list)


def test_legacy_simulation_cannot_validate_code():
    from cad_env.code_generator import CodeExecutor

    executor = CodeExecutor()
    executor.freecad_available = False
    result = executor.execute_code("this is not valid python !!!")
    assert not result["success"] and not result["validated"] and result["simulation"]


def test_nonfinite_rejected_action_is_still_recorded(backend, tmp_path):
    env = CADAgentEnv(tmp_path, backend)
    env.reset(make_task("plate", 1, "edit"))
    observation, reward, done, truncated, info = env.step(
        {
            "op": "set_parameter",
            "target": "HoleSketch",
            "parameter": "diameter_0",
            "value": float("nan"),
        }
    )
    assert not info["execution_ok"] and not done and not truncated
    assert reward == 0 and "error" in observation
    record = json.loads((env.episode_dir / "trajectory.jsonl").read_text())
    assert "invalid_input_repr" in record["action"]
