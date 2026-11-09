"""Collect real TRAIN student prefixes and natively verify teacher continuations.

This first collector writes research evidence and candidate feedback, not an
approved training package. A failed finish is intercepted before execution:
teachers supervise the decision at the same causal state, without inventing a
post-termination environment transition. No held-out task is collected.
"""

import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cad_env.agent.environment import CADAgentEnv
from cad_env.agent.observations import agent_messages, policy_images, transition

from .factory import jsonl, teacher_transition
from .perception_data import sha
from .recovery_teacher import repair_actions


def collect_one(root, out, entry, policy, student_steps, current_render=False):
    task = json.loads((root / "episodes" / entry["task_id"] / "task.json").read_text())
    if task["split"] != "train" or task["split_group"] != entry["split_group"]:
        raise ValueError("TRAIN lineage mismatch")
    task = copy.deepcopy(task)
    if current_render:
        task["goal_only_observation"] = False
    env = CADAgentEnv(out / "native", visual=True, asset_root=root)
    observation, _ = env.reset(task)
    initial = copy.deepcopy(observation["state"])
    history, evidence = [], []
    stopped_on_finish = False
    for _ in range(min(student_steps, task["max_steps"] - 1)):
        prompt = agent_messages(task, observation, history, visual=True)
        images = policy_images(observation)
        response = policy(prompt, images)
        try:
            action = json.loads(response)
        except ValueError:
            action = {"op": "invalid_json"}
        evidence.append(
            {
                "prompt": prompt,
                "images": images,
                "response": response,
                "observation": copy.deepcopy(observation),
            }
        )
        if isinstance(action, dict) and action.get("op") == "finish":
            stopped_on_finish = True
            break
        observation, _, _, _, info = env.step(action)
        history.append(
            {**transition(action, info, observation), "response": response[:2000]}
        )
        if env.closed:
            break
    corrections = repair_actions(task, initial, observation["state"])
    result = {
        "task_id": task["task_id"],
        "split_group": task["split_group"],
        "student_calls": len(evidence),
        "student_steps": env.attempts,
        "proposed_finish": stopped_on_finish,
        "teacher_actions": corrections,
        "current_render": current_render,
        "verified": False,
        "student_evidence": evidence,
    }
    if env.closed or env.attempts + len(corrections) > task["max_steps"]:
        result["rejected"] = "Insufficient remaining native step budget"
        return result
    # A correct finish needs no repair; retain evidence without upweighting it.
    if stopped_on_finish and corrections == [{"op": "finish"}]:
        result["already_correct"] = True
        return result
    rows = []
    for action in corrections:
        rows.append(
            {
                "task_id": task["task_id"],
                "split_group": task["split_group"],
                "step_id": env.attempts,
                "prompt": agent_messages(task, observation, history, visual=True),
                "completion": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(action)}],
                    }
                ],
                "images": policy_images(observation),
            }
        )
        observation, _, _, _, info = env.step(action)
        history.append(teacher_transition(action, info, observation))
        if not info.get("execution_ok"):
            result["rejected"] = "Teacher native execution failed"
            return result
    result["verified"] = bool(info.get("validation", {}).get("task_success"))
    result["validation"] = info.get("validation", {})
    result["trajectory"] = str(env.episode_dir / "trajectory.jsonl")
    result["trajectory_sha256"] = sha(result["trajectory"])
    if result["verified"]:
        result["candidate_feedback"] = rows
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ("dataset", "model", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--parents", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--student-steps", type=int, default=8)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--current-render", action="store_true")
    p.add_argument("--phase", choices=["clean", "revisit"])
    a = p.parse_args()
    if min(a.parents, a.workers, a.student_steps, a.minutes) <= 0:
        p.error("Positive limits required")
    root, out = Path(a.dataset).resolve(), Path(a.output).resolve()
    from .train import check_training_source

    check_training_source(root, root / "exports/train.feedback.jsonl")
    entries, parents = [], set()
    for entry in sorted(
        map(json.loads, (root / "manifest.jsonl").read_text().splitlines()),
        key=lambda e: e["task_id"],
    ):
        if (
            not entry["accepted"]
            or entry["split"] != "train"
            or entry["split_group"] in parents
        ):
            continue
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        if a.phase and task.get("curriculum_phase") != a.phase:
            continue
        if (
            task["mode"] != "edit"
            or len(task.get("goal_spec", {}).get("selected", [])) < 4
        ):
            continue
        entries.append(entry)
        parents.add(entry["split_group"])
        if len(entries) == a.parents:
            break
    if not entries:
        raise ValueError("No eligible multi-edit TRAIN tasks")
    out.mkdir(parents=True, exist_ok=False)
    from .batch_policy import BatchedHFPolicy, PolicyBatcher

    model = BatchedHFPolicy(
        a.model,
        temperature=0,
        max_new_tokens=256,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    batcher = PolicyBatcher(model, a.workers)
    state = {
        "status": "running",
        "train_only": True,
        "test_opened": False,
        "approved_training_package": False,
        "model_identity": model.identity,
        "source_summary_sha256": sha(root / "summary.json"),
        "episodes": [],
        "scope": "Real current-student prefixes; candidate feedback only after native terminal success; not a benchmark or approved training export",
    }

    def persist():
        (out / "status.json").write_text(json.dumps(state, indent=2))

    persist()
    deadline = time.monotonic() + a.minutes * 60
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for start in range(0, len(entries), a.workers):
                if time.monotonic() > deadline:
                    state["budget_exhausted"] = True
                    break
                futures = [
                    pool.submit(
                        collect_one,
                        root,
                        out,
                        e,
                        batcher,
                        a.student_steps,
                        a.current_render,
                    )
                    for e in entries[start : start + a.workers]
                ]
                for future in futures:
                    record = future.result()
                    jsonl(out / (record["task_id"] + ".jsonl"), [record])
                    state["episodes"].append(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("student_evidence", "candidate_feedback")
                        }
                    )
                    persist()
        state["status"] = "complete"
        state["verified_recoveries"] = sum(r["verified"] for r in state["episodes"])
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        batcher.close()
        state["batch_stats"] = batcher.stats
        persist()


if __name__ == "__main__":
    main()
