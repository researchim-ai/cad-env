"""Bounded native CAD-agent pilot: matched ISO/GOST inputs on unseen validation parents."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages, policy_images, transition
from cad_env.drawings.batch import write_json
from cad_env.drawings.dataset import audit_technical_dataset
from cad_env.drawings.export import sha
from cad_env.learning.batch_policy import BatchedHFPolicy, PolicyBatcher


def select_pairs(root, count=20, seed=20261109):
    rows = [json.loads(s) for s in (root / "pairs.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            groups[row["family"], row["difficulty"]].append(row)
    selected = []
    for index, key in enumerate(sorted(groups)):
        ordered = sorted(
            groups[key],
            key=lambda r: hashlib.sha256(f"{seed}:{r['task_id']}".encode()).hexdigest(),
        )
        quota = count // len(groups) + (index < count % len(groups))
        if len(ordered) < quota:
            raise ValueError("Not enough prespecified validation cases")
        selected.extend(ordered[:quota])
    if len(selected) != count or len({r["split_group"] for r in selected}) != count:
        raise ValueError("Expected distinct naturally held-out parents")
    return selected


def summarize(rows):
    result = {}
    for standard in ("iso", "gost"):
        cases = [r for r in rows if r["standard"] == standard]
        valid = [r for r in cases if r["status"] == "complete"]
        result[standard] = {
            "planned": len(cases),
            "evaluated": len(valid),
            "infrastructure_errors": len(cases) - len(valid),
            "successes": sum(r["task_success"] for r in valid),
            "success_rate": sum(r["task_success"] for r in valid) / len(valid)
            if valid
            else None,
            "mean_reward": sum(r["reward"] for r in valid) / len(valid)
            if valid
            else None,
            "stop_reasons": dict(Counter(r["stop_reason"] for r in valid)),
            "total_steps": sum(r["steps"] for r in valid),
            "invalid_actions": sum(r["invalid_actions"] for r in valid),
        }
    return result


def evaluate_case(pair, standard, batcher, out, stagnant_limit):
    representation = pair["representations"][standard]
    dataset = Path(representation["dataset"])
    task = json.loads(Path(representation["task"]).read_text())
    if task["split"] != "validation":
        raise ValueError("Only development validation is allowed")
    case = out / standard / task["task_id"]
    case.mkdir(parents=True, exist_ok=False)
    backend = FreeCADBackend(timeout=180)
    env = CADAgentEnv(case / "native", backend, visual=True, asset_root=dataset)
    observation, _ = env.reset(task)
    history, invalid, stagnant = [], 0, 0
    reward, info, truncated = 0.0, {}, False
    start = time.monotonic()
    stop = "action_budget"
    while not env.closed:
        messages = agent_messages(task, observation, history, visual=True)
        paths = policy_images(observation)
        response = batcher(messages, paths)
        with (case / "responses.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": env.attempts,
                        "response": response,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(messages, sort_keys=True).encode()
                        ).hexdigest(),
                        "image_sha256": [sha(p) for p in paths],
                    }
                )
                + "\n"
            )
        try:
            action = json.loads(response)
        except ValueError:
            action = {"op": "invalid_json"}
        before = json.dumps(observation["state"], sort_keys=True)
        observation, reward, done, truncated, info = env.step(action)
        if info.get("error_type") in ("Timeout", "WorkerCrash", "InfrastructureError"):
            raise RuntimeError(
                "Native execution infrastructure failure: "
                + str(observation.get("error"))
            )
        invalid += int(not info["execution_ok"])
        history.append(
            {**transition(action, info, observation), "response": response[:2000]}
        )
        stagnant = (
            stagnant + 1
            if json.dumps(observation["state"], sort_keys=True) == before
            else 0
        )
        if done:
            stop = "agent_finish"
        elif truncated:
            stop = "action_budget"
        elif stagnant >= stagnant_limit:
            stop = "stagnation_guard"
            break
    # Score the actual student program even if the agent failed to finish. This
    # evaluator-only finish never appears in the policy history or counts as an agent finish.
    if stop == "agent_finish":
        validation = info.get("validation", {})
    else:
        final = backend.run(
            env.actions + [{"op": "finish"}],
            case / "final-score",
            initial_actions=task["initial_actions"],
            reference_actions=task["reference_actions"],
            tolerance=task["tolerance"],
            expected_metrics=task["analytic_metrics"],
            reward_mode="strict_success",
        )
        if final.get("error_type") in ("Timeout", "WorkerCrash", "InfrastructureError"):
            raise RuntimeError(
                "Final native scoring infrastructure failure: "
                + str(final.get("error"))
            )
        validation = final.get("validation", {})
        if (
            validation.get("evaluator_error")
            or final.get("error_origin") == "evaluation"
        ):
            raise RuntimeError("Native evaluator failed: " + str(final.get("error")))
        reward = validation.get("reward", 0.0)
    row = {
        "task_id": task["task_id"],
        "split_group": task["split_group"],
        "standard": standard,
        "family": pair["family"],
        "difficulty": pair["difficulty"],
        "status": "complete",
        "stop_reason": stop,
        "steps": env.attempts,
        "invalid_actions": invalid,
        "reward": reward,
        "task_success": stop == "agent_finish" and bool(validation.get("task_success")),
        "geometry_success_on_final_probe": bool(validation.get("task_success")),
        "validation": validation,
        "elapsed_seconds": time.monotonic() - start,
        "episode_dir": str(env.episode_dir),
        "committed_actions": env.actions,
    }
    write_json(case / "result.json", row)
    print(
        f"{standard} {task['task_id']} {stop} success={row['task_success']} steps={env.attempts}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model", action="append", required=True, help="label=local_checkpoint"
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stagnant-limit", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Use physical GPU1 only")
    if (
        min(
            args.count,
            args.batch_size,
            args.workers,
            args.stagnant_limit,
            args.max_new_tokens,
        )
        < 1
    ):
        parser.error("Positive counts required")
    root, out = Path(args.paired).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = select_pairs(root, args.count)
    write_json(out / "selection.json", selected)
    shutil.copyfile(__file__, out / "runner.py")
    protocol = {
        "physical_gpu": 1,
        "count_by_geometry": args.count,
        "profiles": ["iso", "gost"],
        "split": "validation",
        "selection_seed": 20261109,
        "temperature": 0,
        "max_context": 32768,
        "max_pixels_per_image": 327680,
        "max_new_tokens_per_action": args.max_new_tokens,
        "maximum_action_budget": 128,
        "stagnation_guard": args.stagnant_limit,
        "scope": "Bounded native agent pilot; guard stops consecutive unchanged public states. Not a full-corpus or industrial benchmark.",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "selection_sha256": sha(out / "selection.json"),
        "runner_sha256": sha(__file__),
    }
    status = {
        "status": "auditing",
        "protocol": protocol,
        "models": {},
        "training_started": False,
    }
    write_json(out / "status.json", status)
    datasets = sorted(
        {p["representations"][s]["dataset"] for p in selected for s in ("iso", "gost")}
    )
    for dataset in datasets:
        audit = audit_technical_dataset(dataset)
        if not audit["ok"]:
            raise ValueError(
                "Selected source corpus failed audit: " + str(audit["errors"][:3])
            )
    import torch

    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Required RTX3090 GPU1 is not accessible")
    status["device"] = torch.cuda.get_device_name(0)
    for spec in args.model:
        label, model = spec.split("=", 1)
        model_out = out / label
        model_out.mkdir()
        status["status"] = "loading_model"
        status["models"][label] = {"path": model, "status": "loading", "episodes": []}
        write_json(out / "status.json", status)
        policy = BatchedHFPolicy(
            model,
            max_context=32768,
            max_pixels=327680,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
            chat_template_kwargs={"enable_thinking": False},
        )
        status["models"][label]["identity"] = policy.identity
        batcher = PolicyBatcher(policy, batch_size=args.batch_size)
        rows = []
        status["status"] = "evaluating"
        status["models"][label]["status"] = "evaluating"
        write_json(out / "status.json", status)
        torch.cuda.reset_peak_memory_stats()
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        evaluate_case, p, s, batcher, model_out, args.stagnant_limit
                    ): (p, s)
                    for p in selected
                    for s in ("iso", "gost")
                }
                for future in as_completed(futures):
                    pair, standard = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001 - never count infrastructure failures as model zeroes
                        row = {
                            "task_id": pair["task_id"],
                            "standard": standard,
                            "status": "infrastructure_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print("INFRASTRUCTURE ERROR", row, flush=True)
                    rows.append(row)
                    status["models"][label]["episodes"] = rows
                    status["models"][label]["metrics"] = summarize(rows)
                    write_json(out / "status.json", status)
        finally:
            batcher.close()
        result = {
            "status": "complete"
            if all(r["status"] == "complete" for r in rows)
            else "failed",
            "model_identity": policy.identity,
            "protocol": protocol,
            "episodes": rows,
            "metrics": summarize(rows),
            "batch_stats": batcher.stats,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        write_json(model_out / "evaluation.json", result)
        status["models"][label]["status"] = result["status"]
        status["models"][label]["memory"] = {
            k: result[k] for k in ("peak_allocated_gib", "peak_reserved_gib")
        }
        write_json(out / "status.json", status)
        del futures, batcher, policy
        gc.collect()
        torch.cuda.empty_cache()
    status["status"] = (
        "complete"
        if all(m["status"] == "complete" for m in status["models"].values())
        else "failed"
    )
    write_json(out / "status.json", status)
    print(json.dumps({k: v["metrics"] for k, v in status["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
