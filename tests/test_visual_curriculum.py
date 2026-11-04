import copy
import json

import pytest

from cad_env.agent import FreeCADBackend
from cad_env.agent.observations import agent_messages
from cad_env.agent.tasks import expected_metrics, public_task
from cad_env.learning.complex_tasks import make_complex_task
from cad_env.learning.rewards import score_completion
from cad_env.learning.visual_tasks import make_visual_group, render_goal


@pytest.fixture(scope="module")
def backend():
    try:
        return FreeCADBackend(timeout=30)
    except RuntimeError:
        pytest.skip("Native FreeCAD required")


def test_visual_counterfactuals_hide_choice_and_preserve_aggregate_metrics():
    tasks = make_visual_group(11, selected_count=2)
    assert len({json.dumps(t["initial_actions"], sort_keys=True) for t in tasks}) == 1
    assert len({json.dumps(public_task(t), sort_keys=True) for t in tasks}) == 1
    assert len({json.dumps(expected_metrics(t), sort_keys=True) for t in tasks}) == 1
    assert len({json.dumps(t["oracle_actions"], sort_keys=True) for t in tasks}) == 4
    for task in tasks:
        text = json.dumps(
            agent_messages(
                task, {"state": {}, "goal_images": ["private-path.png"]}, visual=True
            )
        )
        assert (
            task["task_id"] not in text
            and "goal_spec" not in text
            and "private-path" not in text
        )


def test_real_visual_reward_rejects_noop_wrong_target_and_incomplete(tmp_path, backend):
    tasks = make_visual_group(11, selected_count=2)
    task = tasks[0]
    cases = [
        (task["oracle_actions"], True),
        ([{"op": "finish"}], False),
        (tasks[1]["oracle_actions"], False),
        ([task["oracle_actions"][0], {"op": "finish"}], False),
        (list(reversed(task["oracle_actions"][:-1])) + [{"op": "finish"}], True),
    ]
    for i, (actions, expected) in enumerate(cases):
        result = score_completion(task, json.dumps(actions), tmp_path / str(i), backend)
        assert result["task_success"] == expected
        assert result["reward"] == int(expected)


@pytest.mark.parametrize(
    "family,tier",
    [
        ("perforated_plate", "iid"),
        ("boss_plate", "iid"),
        ("stacked_shaft", "iid"),
        ("stacked_shaft", "compositional"),
        ("mixed_boss_plate", "compositional"),
    ],
)
def test_compositional_teachers_are_native_analytic_and_parametric(
    tmp_path, backend, family, tier
):
    task = make_complex_task(family, 7, "image", tier)
    result = score_completion(
        task, json.dumps(task["oracle_actions"]), tmp_path / family, backend
    )
    assert result["task_success"], result
    assert result["validation"]["parametric"]
    assert result["validation"]["spec_pass"]
    assert len(task["oracle_actions"]) >= 8


def test_text_image_conditions_share_lineage_but_image_prompt_has_no_numbers():
    a = make_complex_task("boss_plate", 11, "image")
    b = make_complex_task("boss_plate", 11, "text")
    assert a["split_group"] == b["split_group"] and a["split"] == b["split"]
    assert a["reference_actions"] == b["reference_actions"]
    assert a["task_id"] != b["task_id"]
    assert not any(c.isdigit() for c in public_task(a)["instruction"])


def test_rendered_goal_marker_changes_pixels_without_mutating_initial(tmp_path):
    pytest.importorskip("PIL")
    a, b = make_visual_group(13)[:2]
    initial = copy.deepcopy(a["initial_actions"])
    render_goal(a, tmp_path / "a.png")
    render_goal(b, tmp_path / "b.png")
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()
    assert a["initial_actions"] == initial


def test_variable_units_preserve_parent_split_and_change_native_target(
    tmp_path, backend
):
    a = make_visual_group(11, target_diameter=10)[0]
    b = make_visual_group(11, target_diameter=12.7, public_diameter="0.5 inches")[0]
    assert a["split_group"] == b["split_group"] and a["split"] == b["split"]
    assert a["task_id"] != b["task_id"]
    assert "0.5 inches" in public_task(b)["instruction"]
    assert b["oracle_actions"][0]["value"] == 12.7
    result = score_completion(
        b, json.dumps(b["oracle_actions"]), tmp_path / "unit", backend
    )
    assert result["task_success"] and result["reward"] == 1


def test_control_cache_keys_public_content_and_rejects_sampling(tmp_path):
    from cad_env.learning.visual_controls import GreedyPublicCache

    class Policy:
        temperature = 0

        def __init__(self):
            self.calls = 0

        def __call__(self, messages, images):
            self.calls += 1
            return str(self.calls)

    policy = Policy()
    cache = GreedyPublicCache(policy)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert cache([{"content": "public"}], [a]) == "1"
    assert cache([{"content": "public"}], [b]) == "1"
    b.write_bytes(b"different")
    assert cache([{"content": "public"}], [b]) == "2"
    assert cache.hits == 1 and cache.misses == 2
    policy.temperature = 0.8
    with pytest.raises(ValueError):
        GreedyPublicCache(policy)


def test_wrong_goal_diagnostic_uses_peer_without_replacing_original_reward(
    tmp_path, backend
):
    from cad_env.learning.peer_control import regrade_peer

    root = tmp_path / "dataset"
    tasks = make_visual_group(11)
    for i, task in enumerate(tasks):
        task["goal_images"] = [f"goal-{i}.png"]
        task["counterfactual_goal_images"] = [f"goal-{(i + 1) % 4}.png"]
        folder = root / "episodes" / task["task_id"]
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps(task))
    (root / "manifest.jsonl").write_text(
        "\n".join(
            json.dumps({"accepted": True, "task_id": t["task_id"]}) for t in tasks
        )
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "trajectory.jsonl").write_text(
        "\n".join(
            json.dumps({"action": a, "info": {"execution_ok": True}})
            for a in tasks[1]["oracle_actions"]
        )
    )
    report = {
        "status": "complete",
        "protocol": {"image_condition": "wrong_goal"},
        "episodes": [
            {
                "task_id": tasks[0]["task_id"],
                "trajectory": "trajectory.jsonl",
                "task_success": False,
            }
        ],
    }
    source = run / "evaluation.json"
    source.write_text(json.dumps(report))
    before = source.read_bytes()
    result = regrade_peer(root, source, tmp_path / "diagnostic", workers=1)
    assert result["original_success"] == 0 and result["peer_success"] == 1
    assert source.read_bytes() == before


def test_visual_multiturn_replay_preserves_goal_and_train_boundary(tmp_path, backend):
    from cad_env.learning.multiturn import replay_conversation

    task = make_visual_group(11, force_split="train")[0]
    folder = tmp_path / "dataset" / "episodes" / task["task_id"]
    folder.mkdir(parents=True)
    task["goal_images"] = ["goal.png"]
    render_goal(task, folder.parent.parent / "goal.png")
    (folder / "task.json").write_text(json.dumps(task))
    messages = [
        {"role": "assistant", "content": json.dumps(a)} for a in task["oracle_actions"]
    ]
    result = replay_conversation(
        folder.parent, task["task_id"], messages, tmp_path / "replay", backend=backend
    )
    assert result["task_success"] and result["reward"] == 1
    assert result["observation"]["goal_images"] and not result["observation"]["images"]
    task["split"] = "test"
    (folder / "task.json").write_text(json.dumps(task))
    with pytest.raises(ValueError, match="held-out"):
        replay_conversation(
            folder.parent,
            task["task_id"],
            messages,
            tmp_path / "heldout",
            backend=backend,
        )


def test_bandit_reward_appends_public_finish_and_rejects_heldout(tmp_path):
    from cad_env.learning.visual_bandit import VisualBanditReward

    tasks = make_visual_group(11, force_split="train")
    root = tmp_path / "data"
    for task in tasks:
        folder = root / "episodes" / task["task_id"]
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps(task))
    reward = VisualBanditReward(root, tmp_path / "reward", workers=1)
    correct = reward.score(
        tasks[0]["task_id"], json.dumps(tasks[0]["oracle_actions"][0])
    )
    wrong = reward.score(tasks[0]["task_id"], json.dumps(tasks[1]["oracle_actions"][0]))
    noop = reward.score(tasks[0]["task_id"], '{"op":"finish"}')
    assert correct["reward"] == 1 and correct["task_success"]
    assert wrong["reward"] == 0 and noop["reward"] == 0
    assert wrong["execution_ok"] and not noop["execution_ok"]
    task = tasks[0]
    task["split"] = "validation"
    (root / "episodes" / task["task_id"] / "task.json").write_text(json.dumps(task))
    with pytest.raises(ValueError, match="held-out"):
        reward.score(task["task_id"], json.dumps(task["oracle_actions"][0]))


def test_factory_never_exports_unconditioned_answers_for_image_goals(tmp_path):
    from cad_env.learning.audit import audit_dataset
    from cad_env.learning.factory import generate_dataset
    from cad_env.learning.multiturn import export_swift_tasks

    tasks = make_visual_group(11, force_split="train")
    root = tmp_path / "visual"
    summary = generate_dataset(
        root,
        task_specs=tasks,
        families=("visual_grounding",),
        modes=("edit",),
        preferences=False,
        workers=2,
    )
    assert summary["accepted"] == 4 and summary["rejected"] == 0
    for kind in ("sft", "steps", "grpo", "preference"):
        assert summary["counts"]["train." + kind] == 0
        assert not (root / "exports" / ("train." + kind + ".jsonl")).read_text()
    assert summary["counts"]["train.feedback"] == 8
    feedback = [
        json.loads(line)
        for line in (root / "exports/train.feedback.jsonl").read_text().splitlines()
    ]
    first = feedback[0]
    following = next(
        r for r in feedback if r["task_id"] == first["task_id"] and r["step_id"] == 1
    )
    context, _ = json.JSONDecoder().raw_decode(
        following["prompt"][-1]["content"][0]["text"]
    )
    assert (
        context["recent_transitions"][-1]["response"]
        == first["completion"][0]["content"][0]["text"]
    )
    assert audit_dataset(root)["ok"]
    output = tmp_path / "swift.jsonl"
    assert export_swift_tasks(root, output)["rows"] == 4
    for line in output.read_text().splitlines():
        row = json.loads(line)
        assert len(row["images"]) == 1
        assert "TASK GOAL" in row["messages"][-1]["content"]
        assert "reference_actions" not in json.dumps(row["messages"])


def test_shuffled_teachers_and_entity_labels_preserve_targets_and_public_controls(
    tmp_path,
):
    ascending = make_visual_group(11, selected_count=2)
    shuffled = make_visual_group(11, selected_count=2, shuffle_edit_order=True)
    assert any(
        a["oracle_actions"] != b["oracle_actions"] for a, b in zip(ascending, shuffled)
    )
    for a, b in zip(ascending, shuffled):
        assert a["reference_actions"] == b["reference_actions"]
        assert a["split_group"] == b["split_group"] and public_task(a) == public_task(b)
    labelled = make_visual_group(11, selected_count=2, entity_labels=True)
    assert len({json.dumps(public_task(t), sort_keys=True) for t in labelled}) == 1
    for a, b in zip(ascending, labelled):
        assert a["reference_actions"] == b["reference_actions"]
        assert a["task_id"] != b["task_id"] and a["split_group"] == b["split_group"]
    render_goal(ascending[0], tmp_path / "plain.png")
    render_goal(labelled[0], tmp_path / "labelled.png")
    assert (tmp_path / "plain.png").read_bytes() != (
        tmp_path / "labelled.png"
    ).read_bytes()


def test_http_goal_images_and_native_rewards_are_public_and_session_scoped(tmp_path):
    pytest.importorskip("fastapi")
    import asyncio

    import httpx

    from cad_env.agent.api import create_app
    from cad_env.learning.factory import generate_dataset

    train = make_visual_group(11, force_split="train")
    heldout = make_visual_group(13, force_split="validation")
    root = tmp_path / "data"
    generate_dataset(
        root,
        task_specs=train + heldout,
        families=("visual_grounding",),
        modes=("edit",),
        workers=2,
    )
    app = create_app(
        tmp_path / "sessions", dataset_dir=root, visual=True, train_only=True
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/dataset/sessions", json={"task_id": train[0]["task_id"]}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["messages"] == agent_messages(
                train[0], body["observation"], visual=True
            )
            assert (
                body["images"]
                == body["observation"]["goal_images"] + body["observation"]["images"]
            )
            assert train[0]["task_id"] not in json.dumps(body)
            assert "reference_actions" not in json.dumps(body)
            sid = body["session_id"]
            image_url = body["observation"]["goal_images"][0]
            image = await client.get(image_url)
            assert (
                image.status_code == 200
                and image.headers["content-type"] == "image/png"
            )
            assert (
                image.content
                == (root / "episodes" / train[0]["task_id"] / "goal.png").read_bytes()
            )
            assert (
                await client.get(f"/sessions/{sid}/images/private-reference")
            ).status_code == 404
            assert (
                await client.post(
                    "/dataset/sessions", json={"task_id": "../../private"}
                )
            ).status_code == 404
            assert (
                await client.post(
                    "/dataset/sessions", json={"task_id": heldout[0]["task_id"]}
                )
            ).status_code == 403
            for action in train[0]["oracle_actions"]:
                response = await client.post(
                    f"/sessions/{sid}/step", json={"action": action}
                )
                assert response.status_code == 200
                result = response.json()
                assert result["observation"]["goal_images"][0] == image_url
            assert result["terminated"] and result["reward"] == 1
            assert (await client.delete(f"/sessions/{sid}")).status_code == 200
            assert (await client.get(image_url)).status_code == 404

    asyncio.run(run())


def test_real_policy_prefix_repairs_share_corruption_without_target_leak(tmp_path):
    import hashlib

    from cad_env.learning.factory import generate_dataset
    from cad_env.learning.onpolicy_repairs import generate_repairs
    from cad_env.learning.train import check_training_source

    tasks = make_visual_group(11, force_split="train")
    root = tmp_path / "source"
    generate_dataset(
        root,
        task_specs=tasks,
        families=("visual_grounding",),
        modes=("edit",),
        workers=2,
    )
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "calibration.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "model_identity": {"test_fixture": True},
                "dataset_sha256": hashlib.sha256(
                    (root / "summary.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    # This fixture deliberately supplies another target's action. The generator
    # independently verifies the error in FreeCAD before using it.
    (calibration / "scores.jsonl").write_text(
        json.dumps(
            {
                "task_id": tasks[0]["task_id"],
                "candidate": 0,
                "task_success": False,
                "execution_ok": True,
                "reward": 0,
                "completion": json.dumps(tasks[1]["oracle_actions"][0]),
            }
        )
        + "\n"
    )
    output = tmp_path / "repairs"
    report = generate_repairs(root, calibration, output, workers=2)
    assert report["tasks"] == 4 and report["train_only"]
    assert report["nonvisual_ceiling"] == 0.25
    repaired = [
        json.loads(p.read_text()) for p in (output / "episodes").glob("*/task.json")
    ]
    assert (
        len({json.dumps(t["initial_actions"], sort_keys=True) for t in repaired}) == 1
    )
    assert len({json.dumps(t["public_task"], sort_keys=True) for t in repaired}) == 1
    assert all(
        t["split"] == "train" and t["family"] == "visual_repair" for t in repaired
    )
    assert any(t["oracle_actions"] == [{"op": "finish"}] for t in repaired)
    assert any(len(t["oracle_actions"]) == 3 for t in repaired)
    assert check_training_source(output, output / "exports/train.feedback.jsonl")["ok"]


def test_diagnostics_keep_invalid_action_containers_in_denominators(tmp_path):
    from cad_env.learning.curriculum_metrics import summarize_curriculum

    task = make_complex_task("stacked_shaft", 7, "image")
    root = tmp_path / "data"
    folder = root / "episodes" / task["task_id"]
    folder.mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps(task))
    run = tmp_path / "run"
    run.mkdir()
    (run / "trajectory.jsonl").write_text(
        json.dumps({"action": [], "info": {"execution_ok": False}}) + "\n"
    )
    report = {
        "status": "complete",
        "episodes": [
            {
                "task_id": task["task_id"],
                "trajectory": "trajectory.jsonl",
                "task_success": False,
                "reward": 0,
            }
        ],
    }
    path = run / "evaluation.json"
    path.write_text(json.dumps(report))
    result = summarize_curriculum(root, path)
    assert result["first_action_execution_validity"] == 0
    assert result["initial_primitive_parameter_accuracy"] == 0
    assert result["slices"]["input:image"]["tasks"] == 1
    assert result["slices"]["input:image"]["success"] == 0
