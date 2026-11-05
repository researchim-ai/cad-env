"""Held-out one-action evaluation matched to the actual visual-bandit RL objective."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import set_seed

from cad_env.agent.observations import normalize_single_action_prompt
from cad_env.agent.schema import validate_action

from .policy import HFPolicy
from .rewards import score_completion
from .task_identity import load_private_task
from .visual_audit import audit_visual


def main():
    p = argparse.ArgumentParser(__doc__)
    for arg in ["dataset", "model", "output"]:
        p.add_argument("--" + arg, required=True)
    args = p.parse_args()
    root = Path(args.dataset).resolve()
    out = Path(args.output).resolve()
    audit = audit_visual(root)
    if not audit["ok"]:
        raise ValueError("Visual corpus audit failed")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh matched evaluation")
    rows = []
    tasks = {}
    for line in (root / "exports/validation.feedback.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["step_id"] != 0:
            continue
        task = load_private_task(root / "episodes", r["task_id"])
        if task["split"] != "validation":
            raise ValueError("Unexpected split")
        if (
            task["family"] == "visual_grounding"
            and len(task["goal_spec"]["selected"]) == 1
        ):
            rows.append(r)
            tasks[r["task_id"]] = task
    if not rows:
        raise ValueError("No held-out single-hole prompts")
    out.mkdir(parents=True)
    policy = HFPolicy(
        args.model,
        temperature=0,
        max_new_tokens=128,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    eos = policy.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if policy.tokenizer.eos_token_id not in eos:
        eos = [policy.tokenizer.eos_token_id, *eos]
    policy.model.generation_config.eos_token_id = eos
    protocol = {
        "scope": "one sampled set_parameter then fixed public finish",
        "temperature": 0.0,
        "max_new_tokens": 128,
        "max_context": 8192,
        "max_pixels": 327680,
        "eos_token_ids": eos,
        "split": "validation",
        "selected_count": 1,
        "prompt_contract": "normalized single JSON action",
    }
    report = {
        "status": "running",
        "limited_evaluation": False,
        "dataset_digest": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
        "protocol": protocol,
        "protocol_digest": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "model_identity": policy.identity,
        "task_ids": sorted(r["task_id"] for r in rows),
        "episodes": [],
    }

    def persist():
        (out / "evaluation.json").write_text(json.dumps(report, indent=2))

    persist()
    try:
        for row in rows:
            task = tasks[row["task_id"]]
            set_seed(42)
            images = []
            for relative in row["images"]:
                image = (root / relative).resolve()
                if not image.is_relative_to(root):
                    raise ValueError("Image escaped corpus")
                images.append(str(image))
            completion = policy(normalize_single_action_prompt(row["prompt"]), images)
            try:
                action = validate_action(json.loads(completion))
                if action["op"] != "set_parameter":
                    raise ValueError("Requires one parameter edit")
            except (ValueError, TypeError):
                result = {"execution_ok": False, "task_success": False, "reward": 0.0}
            else:
                result = score_completion(
                    task,
                    json.dumps([action, {"op": "finish"}]),
                    out / "episodes" / row["task_id"],
                )
            report["episodes"].append(
                {
                    "task_id": row["task_id"],
                    "split_group": task["split_group"],
                    "attempt": 0,
                    "completion": completion,
                    "execution_ok": result["execution_ok"],
                    "task_success": result["task_success"],
                    "reward": result["reward"],
                }
            )
            persist()
        report.update(
            status="complete",
            metrics={
                "tasks": len(rows),
                "success_at_1": sum(r["task_success"] for r in report["episodes"])
                / len(rows),
                "independent_groups": len(
                    {r["split_group"] for r in report["episodes"]}
                ),
            },
        )
        persist()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        persist()
        raise


if __name__ == "__main__":
    main()
