"""Quality gates, truthful evaluation, visual feedback and reward invariants."""

import json
from pathlib import Path

import pytest

from cad_env.agent import FreeCADBackend, make_task
from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages
from cad_env.learning.factory import generate_dataset
from cad_env.learning.metrics import paired_comparison, summarize_episodes
from cad_env.learning.protocol import evaluate_policy
from cad_env.learning.rewards import score_completion
from cad_env.learning.train import check_training_source
from cad_env.learning.workflow import dataset_readiness, grpo_readiness, training_plan


def test_single_action_prompt_migration_has_no_contradictory_contract():
    from cad_env.agent.observations import (
        agent_messages,
        normalize_single_action_prompt,
    )
    from cad_env.agent.schema import ACTION_GUIDE
    from cad_env.agent.tasks import make_task

    old = [{"role": "system", "content": [{"type": "text", "text": ACTION_GUIDE}]}]
    migrated = normalize_single_action_prompt(old)
    assert "JSON array" in old[0]["content"][0]["text"]
    assert "JSON array" not in migrated[0]["content"][0]["text"]
    assert normalize_single_action_prompt(migrated) == migrated
    assert '"op":"pad"' in migrated[0]["content"][0]["text"]
    assert "JSON array" not in str(agent_messages(make_task(), {}))


@pytest.fixture(scope="module")
def backend():
    try:
        return FreeCADBackend(timeout=20)
    except RuntimeError:
        pytest.skip("Real FreeCAD is required")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, backend):
    root = tmp_path_factory.mktemp("process") / "data"
    generate_dataset(
        root,
        seeds=3,
        render=True,
        families=("plate", "bushing", "stepped_shaft", "flange"),
        holdout_families=("flange",),
        workers=2,
    )
    return root


def test_visual_observation_is_current_state_and_survives_rejection(tmp_path, backend):
    from PIL import Image

    env = CADAgentEnv(tmp_path, backend, visual=True)
    task = make_task()
    obs, _ = env.reset(task)
    blank = Image.open(obs["images"][0])
    assert blank.getextrema() == ((255, 255), (255, 255), (255, 255))
    env.step(task["oracle_actions"][0])
    obs, *_ = env.step(task["oracle_actions"][1])
    assert Image.open(obs["images"][0]).getextrema() != blank.getextrema()
    image = obs["images"][0]
    obs, *_ = env.step(
        {"op": "set_parameter", "target": "Missing", "parameter": "length", "value": 1}
    )
    assert obs["images"] == [image]
    prompt = agent_messages(task, obs, visual=True)
    assert "reference_actions" not in json.dumps(prompt)
    assert "last_error" in json.dumps(prompt)


def test_feedback_export_has_real_current_images_and_recovery(corpus):
    rows = [
        json.loads(line)
        for line in (corpus / "exports/train.feedback.jsonl").read_text().splitlines()
    ]
    assert rows and any(r["recovery"] for r in rows)
    duplicate_rows = [r for r in rows if r.get("recovery_kind") == "duplicate_name"]
    assert duplicate_rows
    assert all(
        "Duplicate object name" in json.dumps(r["prompt"]) for r in duplicate_rows
    )
    assert all(
        '"op": "pad"' in json.dumps(r["completion"]).replace('\\"', '"')
        for r in duplicate_rows
    )
    assert all(
        len(r["images"]) == 1 and r["observation_kind"] == "current_state" for r in rows
    )
    assert all((corpus / r["images"][0]).is_file() for r in rows)
    assert all("reference_actions" not in json.dumps(r["prompt"]) for r in rows)
    small = dataset_readiness(
        corpus, min_train=1, min_validation=1, min_test=1, min_groups=1
    )
    assert small["ready"], small["errors"]
    production = dataset_readiness(corpus)
    assert not production["ready"]


def test_training_rejects_heldout_export(corpus):
    assert check_training_source(corpus, corpus / "exports/train.feedback.jsonl")["ok"]
    with pytest.raises(ValueError, match="held-out"):
        check_training_source(corpus, corpus / "exports/test.feedback.jsonl")


def test_edit_cannot_change_protected_dimensions_with_equal_final_geometry(
    tmp_path, backend
):
    task = make_task("plate", 1, "edit")
    actions = [
        task["oracle_actions"][0],
        {
            "op": "set_parameter",
            "target": "HolePocket",
            "parameter": "length",
            "value": 100,
        },
        {"op": "finish"},
    ]
    score = score_completion(task, json.dumps(actions), tmp_path, backend)
    assert score["validation"]["geometry_score"] == 1
    assert not score["validation"]["protected_parameters_preserved"]
    assert score["reward"] == 0 and not score["task_success"]


def test_grpo_gate_rejects_zero_signal_and_invalid_policy():
    invalid = [
        {"task_id": str(i), "task_success": False, "execution_ok": False, "reward": 0}
        for i in range(20)
        for _ in range(4)
    ]
    report = grpo_readiness(invalid)
    assert not report["ready"] and report["zero_variance_fraction"] == 1
    varied = []
    for i in range(20):
        varied.extend(
            {
                "task_id": str(i),
                "task_success": j == 0,
                "execution_ok": True,
                "reward": 1 if j == 0 else 0.5,
            }
            for j in range(4)
        )
    assert grpo_readiness(varied)["ready"]


def report(success, groups=25):
    rows = [
        {
            "task_id": f"t-{i}",
            "split_group": f"g-{i}",
            "attempt": 0,
            "task_success": success,
            "family": "plate",
            "mode": "create",
            "reward": float(success),
            "steps": 2,
            "invalid_actions": 0,
            "elapsed_seconds": 1,
            "truncated": False,
        }
        for i in range(groups)
    ]
    return {
        "status": "complete",
        "dataset_digest": "frozen",
        "protocol_digest": "same",
        "task_ids": [r["task_id"] for r in rows],
        "episodes": rows,
        "limited_evaluation": False,
    }


def test_paired_metrics_require_same_complete_contract():
    before, after = report(False), report(True)
    assert summarize_episodes(before["episodes"])["success_at_1"] == 0
    comparison = paired_comparison(before, after, resamples=200)
    assert comparison["success_at_1_delta"] == 1 and comparison["improvement_supported"]
    after["protocol_digest"] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        paired_comparison(before, after)
    after = report(True)
    after["status"] = "failed"
    with pytest.raises(ValueError, match="complete"):
        paired_comparison(before, after)
    assert not paired_comparison(report(False, 3), report(True, 3), resamples=200)[
        "improvement_supported"
    ]


def test_frozen_sequential_evaluation_retains_failure_and_has_no_privileged_inputs(
    corpus, tmp_path, backend
):
    def policy_factory(seed):
        def policy(messages, images):
            assert len(images) == 1
            assert "reference_actions" not in json.dumps(messages)
            return "invalid JSON"

        return policy

    result = evaluate_policy(
        corpus,
        policy_factory,
        tmp_path / "eval",
        backend=backend,
        visual=True,
        max_steps=1,
        attempts=2,
    )
    assert result["status"] == "complete" and result["metrics"]["success_at_1"] == 0
    assert all(r["truncated"] for r in result["episodes"])
    assert result["metrics"]["invalid_action_rate"] == 1
    with pytest.raises(ValueError, match="final_test"):
        evaluate_policy(corpus, policy_factory, tmp_path / "test", split="test")


def test_dual_gpu_plan_matches_profile_and_has_no_implicit_execution(tmp_path):
    recipe = json.loads(Path("recipes/dual-3090-vlm.json").read_text())
    plan = training_plan(
        recipe, tmp_path / "data", tmp_path / "run", smoke=True, python="python"
    )
    command = plan["command"]
    assert "--nproc_per_node=2" in command
    assert command[command.index("--max-steps") + 1] == "10"
    assert command[command.index("--gradient-accumulation") + 1] == "8"
    assert "--require-cuda" in command


def test_multiturn_replay_recovers_but_requires_explicit_finish(
    corpus, tmp_path, backend
):
    from cad_env.learning.multiturn import export_swift_tasks, replay_conversation

    entry = json.loads((corpus / "exports/train.sft.jsonl").read_text().splitlines()[0])
    task = json.loads(
        (corpus / "episodes" / entry["task_id"] / "task.json").read_text()
    )
    messages = [{"role": "assistant", "content": "invalid JSON"}] + [
        {"role": "assistant", "content": json.dumps(a)} for a in task["oracle_actions"]
    ]
    result = replay_conversation(
        corpus / "episodes",
        task["task_id"],
        messages,
        tmp_path / "valid",
        backend=backend,
        visual=False,
    )
    assert (
        result["reward"] == 1
        and result["task_success"]
        and result["invalid_actions"] == 1
    )
    result = replay_conversation(
        corpus / "episodes",
        task["task_id"],
        messages[:-1],
        tmp_path / "unfinished",
        backend=backend,
        visual=False,
    )
    assert not result["finished"] and result["reward"] == 0
    path = tmp_path / "swift.jsonl"
    stats = export_swift_tasks(corpus, path)
    assert stats["rows"] > 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert "reference_actions" not in json.dumps(row)
        assert "<image>" in row["messages"][-1]["content"]


def test_program_grpo_rejects_next_action_checkpoint(tmp_path):
    recipe = json.loads(Path("recipes/dual-3090-vlm.json").read_text())
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "cad-training-run.json").write_text(
        json.dumps({"action_contract": "single_action"})
    )
    with pytest.raises(ValueError, match="next-action"):
        training_plan(
            recipe,
            tmp_path / "data",
            tmp_path / "out",
            method="grpo",
            sft_checkpoint=checkpoint,
        )


def test_dense_vlm_resume_converts_save_keys_without_losing_weights(tmp_path):
    if __import__("importlib.metadata").metadata.version("trl") != "0.24.0":
        pytest.skip("Run ML-interface tests in the pinned .venv-learning environment")
    import torch
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    from cad_env.learning.checkpoints import restore_checkpoint_weights

    config = LlavaConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        ),
        text_config=LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        image_token_index=4,
    )
    model = LlavaForConditionalGeneration(config)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path / "checkpoint")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    result = restore_checkpoint_weights(model, tmp_path / "checkpoint")
    assert result["exact_roundtrip"]
    assert all(
        torch.equal(value, expected[key]) for key, value in model.state_dict().items()
    )


def test_adapter_resume_is_exact(tmp_path):
    if __import__("importlib.metadata").metadata.version("trl") != "0.24.0":
        pytest.skip("Run in the pinned ML environment")
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import GPT2Config, GPT2LMHeadModel

    from cad_env.learning.checkpoints import restore_checkpoint_weights

    model = get_peft_model(
        GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2)),
        LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.normal_()
    expected = {
        k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()
    }
    model.save_pretrained(tmp_path / "adapter")
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    result = restore_checkpoint_weights(model, tmp_path / "adapter")
    assert result["exact_roundtrip"] and result["kind"] == "adapter"
    assert all(
        torch.equal(value, expected[key])
        for key, value in get_peft_model_state_dict(model).items()
    )


def test_onpolicy_bundle_is_immutable_and_keeps_images_within_package(tmp_path):
    from PIL import Image

    from cad_env.learning.bundles import bundle_supervision, verify_bundle

    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "white").save(image)
    root = tmp_path / "bundle"
    rows = [
        {
            "task_id": "plate-000001-edit",
            "prompt": [],
            "completion": [],
            "images": [str(image)],
        }
    ]
    manifest = bundle_supervision(rows, root, "parent-hash")
    assert (
        manifest["rows"] == 1
        and verify_bundle(root, root / "exports/train.feedback.jsonl")["assets"]
    )
    bundled = root / next(iter(manifest["assets"]))
    bundled.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        verify_bundle(root, root / "exports/train.feedback.jsonl")


def test_numerical_guard_refuses_nan_and_zero_signal():
    if __import__("importlib.metadata").metadata.version("trl") != "0.24.0":
        pytest.skip("Pinned ML environment")
    from cad_env.learning.guards import make_numerical_guard

    guard = make_numerical_guard(True, max_zero_steps=2)
    guard.on_log(None, None, None, {"loss": 1.0, "grad_norm": 0.0})
    with pytest.raises(RuntimeError, match="No gradient"):
        guard.on_log(None, None, None, {"loss": 1.0, "grad_norm": 0.0})
    guard = make_numerical_guard(True)
    with pytest.raises(FloatingPointError):
        guard.on_log(None, None, None, {"loss": float("nan")})
    guard.on_log(None, None, None, {"loss": 1.0, "grad_norm": 0.5})
    guard.on_train_end(None, None, None)
    assert guard.summary()["nonzero_steps"] == 1
