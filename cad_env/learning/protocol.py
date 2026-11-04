"""Frozen evaluation protocol; no privileged labels enter policy messages."""

import hashlib
import json
import time
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages, policy_images, transition

from .factory import jsonl
from .metrics import summarize_episodes


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_policy(
    dataset_dir,
    policy_factory,
    output_dir,
    *,
    split="validation",
    attempts=1,
    visual=True,
    seed=42,
    backend=None,
    max_steps=None,
    policy_protocol=None,
    model_identity=None,
    final_test=False,
    limit=None,
    image_condition="full",
):
    if image_condition not in ("full", "no_goal", "wrong_goal", "blank_goal"):
        raise ValueError("Unknown image condition")
    if split not in ("validation", "test") or (split == "test" and not final_test):
        raise ValueError(
            "Use validation for development; test requires final_test=True after checkpoint selection"
        )
    if attempts < 1:
        raise ValueError("attempts must be positive")
    root, out = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output must be new")
    out.mkdir(parents=True, exist_ok=True)
    from .audit import audit_dataset

    audit = audit_dataset(root)
    if not audit["ok"]:
        raise ValueError("Dataset audit failed: " + str(audit["errors"][:3]))
    summary = json.loads((root / "summary.json").read_text())
    backend = backend or FreeCADBackend()
    runtime_probe = backend.run([], out / "runtime-probe")
    if not runtime_probe.get("execution_ok") or not runtime_probe.get("runtime"):
        raise RuntimeError("Cannot establish evaluation FreeCAD runtime")
    if runtime_probe["runtime"] != summary.get("runtime"):
        raise RuntimeError("Evaluation FreeCAD runtime differs from the frozen dataset")
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = sorted(
        (r for r in manifest if r["accepted"] and r["split"] == split),
        key=lambda r: r["task_id"],
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selected evaluation split is empty")
    # Identity deliberately excludes model: baseline and candidate use one contract.
    source_root = Path(__file__).resolve().parents[1]
    contract = {
        "schema_version": "cad-env.eval.v1",
        "latency_scope": "Policy generation and steps; excludes initial reset and checkpoint loading",
        "split": split,
        "attempts": attempts,
        "visual": visual,
        **({"image_condition": image_condition} if image_condition != "full" else {}),
        "seed": seed,
        "max_steps": max_steps,
        "policy_protocol": policy_protocol or {},
        "agent_prompt_sha256": hashlib.sha256(
            (source_root / "agent/observations.py").read_bytes()
        ).hexdigest(),
        "worker_sha256": hashlib.sha256(
            (source_root / "agent/worker.py").read_bytes()
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            (source_root / "agent/schema.py").read_bytes()
        ).hexdigest(),
        "runtime": runtime_probe["runtime"],
    }
    report = {
        "dataset_digest": digest(summary),
        "protocol": contract,
        "protocol_digest": digest(contract),
        "task_ids": [r["task_id"] for r in selected],
        "model_identity": model_identity or {},
        "episodes": [],
        "status": "running",
        "limited_evaluation": limit is not None,
    }
    report_path = out / "evaluation.json"

    def persist():
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        )

    persist()
    try:
        for entry in selected:
            task = json.loads(
                (root / "episodes" / entry["task_id"] / "task.json").read_text()
            )
            if max_steps is not None:
                task["max_steps"] = min(task["max_steps"], max_steps)
            for attempt in range(attempts):
                sample_seed = (
                    seed
                    + int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16)
                    + attempt
                )
                policy = policy_factory(sample_seed)
                env = CADAgentEnv(
                    out / "episodes", backend, visual=visual, asset_root=root
                )
                observation, _ = env.reset(task)
                history = []
                invalid = 0
                elapsed = time.monotonic()
                while not env.closed:
                    policy_observation = dict(observation)
                    if image_condition == "no_goal":
                        policy_observation["goal_images"] = []
                    elif visual and image_condition == "wrong_goal":
                        replacements = task.get("counterfactual_goal_images")
                        if not replacements:
                            raise ValueError(
                                "Wrong-goal control requires a counterfactual peer"
                            )
                        resolved = [(root / p).resolve() for p in replacements]
                        if any(
                            not p.is_relative_to(root) or not p.is_file()
                            for p in resolved
                        ):
                            raise ValueError("Unsafe counterfactual image")
                        policy_observation["goal_images"] = list(map(str, resolved))
                    elif visual and image_condition == "blank_goal":
                        from PIL import Image

                        blank = out / "blank-goal.png"
                        if not blank.exists():
                            Image.new("RGB", (512, 512), "white").save(blank)
                        policy_observation["goal_images"] = [str(blank)] * len(
                            observation.get("goal_images", [])
                        )
                    messages = agent_messages(
                        task, policy_observation, history, visual=visual
                    )
                    input_record = {
                        "step": env.attempts,
                        "prompt_sha256": digest(messages),
                        "image_sha256": [
                            hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in policy_images(policy_observation)
                        ],
                    }
                    with (env.episode_dir / "policy-inputs.jsonl").open("a") as stream:
                        stream.write(json.dumps(input_record) + "\n")
                    # Only public prompt and image paths are passed to a policy.
                    response = policy(messages, policy_images(policy_observation))
                    if not isinstance(response, str):
                        raise TypeError("Policy must return JSON text")
                    try:
                        action = json.loads(response)
                    except ValueError:
                        action = {"op": "invalid_json"}
                    observation, reward, done, truncated, info = env.step(action)
                    history.append(
                        {
                            **transition(action, info, observation),
                            "response": response[:2000],
                        }
                    )
                    invalid += int(not info["execution_ok"])
                episode = {
                    "task_id": task["task_id"],
                    "split_group": task["split_group"],
                    "family": task["family"],
                    "mode": task["mode"],
                    **(
                        {
                            "condition_group": task["condition_group"],
                            "evaluation_tier": task.get("evaluation_tier", "iid"),
                        }
                        if "condition_group" in task
                        else {}
                    ),
                    "attempt": attempt,
                    "seed": sample_seed,
                    "reward": reward,
                    "task_success": bool(
                        info.get("validation", {}).get("task_success")
                    ),
                    "steps": env.attempts,
                    "invalid_actions": invalid,
                    "truncated": truncated,
                    "elapsed_seconds": time.monotonic() - elapsed,
                    "trajectory": str(
                        (env.episode_dir / "trajectory.jsonl").relative_to(out)
                    ),
                }
                report["episodes"].append(episode)
                persist()
        report["metrics"] = summarize_episodes(report["episodes"])
        report["status"] = "complete"
        persist()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        persist()
        raise
    jsonl(out / "episodes.jsonl", report["episodes"])
    return report
