"""Model-independent multi-turn collection and rejection-sampling supervision.

The caller supplies a policy(messages) -> one action JSON object. No credentials,
provider, or paid API is selected implicitly by this module.
"""

import json
from pathlib import Path

from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages, policy_images, transition

from .factory import jsonl
from .rewards import group_statistics


def collect_group(
    task,
    policy,
    output_dir,
    *,
    num_generations=4,
    backend=None,
    visual=False,
    asset_root=None,
):
    if num_generations < 2:
        raise ValueError("Collect at least two rollouts per task for group diagnostics")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory for a rollout group")
    output.mkdir(parents=True, exist_ok=True)
    rows, successful = [], []
    for candidate in range(num_generations):
        env = CADAgentEnv(
            output / "episodes", backend, visual=visual, asset_root=asset_root
        )
        observation, _ = env.reset(task)
        history = []
        decisions = []
        while not env.closed:
            messages = agent_messages(task, observation, history, visual=visual)
            response = (
                policy(messages, policy_images(observation))
                if visual
                else policy(messages)
            )
            if not isinstance(response, str):
                raise TypeError("policy must return JSON text")
            try:
                action = json.loads(response)
            except ValueError:
                action = {"op": "invalid_json"}
            prompt = list(messages)
            prompt_images = policy_images(observation)
            observation, reward, done, truncated, info = env.step(action)
            history.append(
                {**transition(action, info, observation), "response": response[:2000]}
            )
            decisions.append(
                {
                    "task_id": task["task_id"],
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": response}],
                    "accepted_action": info["execution_ok"],
                    **({"images": list(prompt_images)} if visual else {}),
                }
            )
        success = bool(info.get("validation", {}).get("task_success"))
        rows.append(
            {
                "task_id": task["task_id"],
                "candidate": candidate,
                "reward": reward,
                "task_success": success,
                "steps": env.attempts,
                "execution_ok": info.get("execution_ok", False),
                "shape_valid": info.get("validation", {}).get("shape_valid", False),
                "trajectory": str(env.episode_dir / "trajectory.jsonl"),
            }
        )
        if success and task["split"] == "train":
            # Train only on valid decisions. Previous failures remain in context for repair.
            successful.extend(
                {k: v for k, v in row.items() if k != "accepted_action"}
                for row in decisions
                if row["accepted_action"]
            )
    jsonl(output / "rollouts.jsonl", rows)
    jsonl(output / "successful.steps.sft.jsonl", successful)
    if visual and successful:
        from .bundles import bundle_supervision

        bundle_supervision(successful, output)
    report = group_statistics([r["reward"] for r in rows])
    report["split"] = task["split"]
    report["successful_rollouts"] = sum(r["task_success"] for r in rows)
    (output / "group.json").write_text(json.dumps(report, indent=2))
    return report
