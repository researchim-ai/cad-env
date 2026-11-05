"""Full-agent native evaluation for the optional multi-engine tool environment."""

import hashlib
import json
import time
from pathlib import Path

from .dataset import audit
from .environment import MultiCADEnv


def evaluate(
    dataset,
    policy,
    output,
    *,
    split="validation",
    engine=None,
    final_test=False,
    python=None,
    blender=None,
):
    root = Path(dataset).resolve()
    out = Path(output).resolve()
    if split not in {"validation", "test"} or (split == "test" and not final_test):
        raise ValueError("Test requires explicit --final-test")
    report = audit(root)
    if not report["ok"]:
        raise ValueError("Dataset audit failed: " + str(report["errors"][:3]))
    rows = [
        r
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if (r := json.loads(line))["accepted"]
        and r["split"] == split
        and (engine is None or r["engine"] == engine)
    ]
    if not rows:
        raise ValueError("No eligible held-out tasks")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation output")
    out.mkdir(parents=True)
    result = {
        "status": "running",
        "dataset_sha256": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
        "split": split,
        "engine": engine,
        "model_identity": getattr(policy, "identity", {}),
        "protocol": {
            "action_contract": "single_action",
            "max_steps": "task budget",
            "temperature": getattr(policy, "temperature", None),
            "max_new_tokens": getattr(policy, "max_new_tokens", None),
            "max_context": getattr(policy, "max_context", None),
        },
        "episodes": [],
    }

    def persist():
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))

    persist()
    try:
        for row in rows:
            task = json.loads(
                (root / "episodes" / row["task_id"] / "task.json").read_text()
            )
            env = MultiCADEnv(
                row["engine"],
                out / "episodes" / row["task_id"],
                python=python,
                blender=blender,
            )
            observation = env.reset(task, asset_root=root)
            done = False
            reward = 0.0
            invalid = 0
            trajectory = []
            start = time.monotonic()
            while not done:
                response = policy(
                    env.messages(),
                    [*observation["goal_images"], *observation["images"]],
                )
                try:
                    action = json.loads(response)
                except (ValueError, TypeError):
                    action = {"op": "invalid_json"}
                observation, reward, done, info = env.step(action)
                invalid += not info["execution_ok"]
                trajectory.append(
                    {
                        "response": response,
                        "action": action,
                        "execution_ok": info["execution_ok"],
                        "error": info.get("error"),
                    }
                )
            directory = out / "episodes" / row["task_id"]
            (directory / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
            from .metrics import surface_metrics

            candidate_mesh = directory / f"step-{env.steps:03d}" / "mesh.json"
            surface = {"surface_coverage_1pct": 0.0, "normalized_chamfer_l1": None}
            if candidate_mesh.is_file():
                try:
                    surface = surface_metrics(
                        candidate_mesh,
                        root / "episodes" / row["task_id"] / "reference/mesh.json",
                    )
                except ValueError:
                    pass
            result["episodes"].append(
                {
                    "task_id": row["task_id"],
                    "split_group": row["split_group"],
                    "engine": row["engine"],
                    "family": row["family"],
                    "mode": row["mode"],
                    "task_success": reward == 1.0,
                    "reward": reward,
                    "steps": env.steps,
                    "invalid_actions": invalid,
                    "elapsed_seconds": time.monotonic() - start,
                    "surface_diagnostics": surface,
                }
            )
            persist()
        episodes = result["episodes"]
        result.update(
            status="complete",
            metrics={
                "tasks": len(episodes),
                "success_at_1": sum(e["task_success"] for e in episodes)
                / len(episodes),
                "mean_reward": sum(e["reward"] for e in episodes) / len(episodes),
            },
        )
        result["breakdown"] = {}
        for field in ("engine", "family", "mode"):
            result["breakdown"][field] = {
                value: {
                    "tasks": sum(e[field] == value for e in episodes),
                    "success_at_1": sum(
                        e["task_success"] for e in episodes if e[field] == value
                    )
                    / sum(e[field] == value for e in episodes),
                }
                for value in sorted({e[field] for e in episodes})
            }
        persist()
        return result
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        persist()
        raise
