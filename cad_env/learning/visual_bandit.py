"""Native one-action visual grounding curriculum for RL calibration.

The sampled policy action is followed by a fixed, public finish action. This
is a contextual bandit, not multi-turn agent RL. No oracle action is appended.
Only single-marked-hole TRAIN tasks are eligible. Native strict geometry and
parameter checks determine the reward, including preservation of other holes.
"""

import argparse
import hashlib
import json
import random
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cad_env.agent.observations import normalize_single_action_prompt
from cad_env.agent.schema import validate_action

from .factory import jsonl
from .multiturn import message_text
from .rewards import group_statistics, score_completion
from .task_identity import load_private_task


def bandit_prompts(dataset):
    from .train import check_training_source

    root = Path(dataset).resolve()
    source = root / "exports/train.feedback.jsonl"
    check_training_source(root, source)
    rows = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["step_id"] != 0:
            continue
        task = load_private_task(root / "episodes", row["task_id"], require_train=True)
        if (
            task["family"] != "visual_grounding"
            or task.get("reward_mode") != "strict_success"
            or len(task["goal_spec"]["selected"]) != 1
        ):
            continue
        # Deliberately discard expert completions and every later teacher state.
        rows.append(
            {
                "task_id": row["task_id"],
                "prompt": normalize_single_action_prompt(row["prompt"]),
                "images": row["images"],
            }
        )
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Expected unique single-action train prompts")
    return rows


class VisualBanditReward:
    __name__ = "strict_native_visual_bandit"

    def __init__(self, dataset, output, workers=4):
        self.root = Path(dataset).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.group_variance = []

    def score(self, task_id, completion):
        task = load_private_task(self.root / "episodes", task_id, require_train=True)
        if (
            task["family"] != "visual_grounding"
            or len(task["goal_spec"]["selected"]) != 1
        ):
            raise ValueError("This reward only supports single-mark visual grounding")
        if task.get("reward_mode") != "strict_success":
            raise ValueError("Binary native task reward is required")
        directory = self.output / task_id / uuid.uuid4().hex
        directory.mkdir(parents=True)
        try:
            if isinstance(completion, list):
                if len(completion) != 1 or completion[0]["role"] != "assistant":
                    raise ValueError("Expected one assistant action")
                completion = message_text(completion[0]["content"])
            action = validate_action(json.loads(completion))
            if action["op"] != "set_parameter":
                raise ValueError("The one-action curriculum requires a parameter edit")
        except (ValueError, TypeError, KeyError) as error:
            result = {
                "reward": 0.0,
                "task_success": False,
                "execution_ok": False,
                "error_type": "InvalidBanditAction",
                "error": str(error),
            }
        else:
            result = score_completion(
                task, json.dumps([action, {"op": "finish"}]), directory / "native"
            )
        result.update(
            task_id=task_id,
            completion=completion,
            contract="one sampled action then fixed public finish",
        )
        (directory / "score.json").write_text(json.dumps(result, indent=2))
        return result

    def __call__(self, completions, task_id, log_metric=None, **kwargs):
        if len(completions) != len(task_id):
            raise ValueError("Each rollout needs a private task identity")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            scores = list(
                pool.map(lambda pair: self.score(*pair), zip(task_id, completions))
            )
        groups = defaultdict(list)
        for score in scores:
            groups[score["task_id"]].append(score["reward"])
        stats = [group_statistics(v) for v in groups.values() if len(v) >= 2]
        self.group_variance.extend(s["std"] for s in stats)
        if log_metric and stats:
            log_metric(
                "native/mixed_group_fraction",
                sum(not s["zero_variance"] for s in stats) / len(stats),
            )
        return [s["reward"] for s in scores]


def calibrate(dataset, model, output, prompts=20, generations=4, seed=42):
    from transformers import set_seed

    from .policy import HFPolicy
    from .workflow import grpo_readiness

    root, out = Path(dataset).resolve(), Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new calibration output")
    out.mkdir(parents=True, exist_ok=True)
    rows = bandit_prompts(root)
    random.Random(seed).shuffle(rows)
    rows = rows[:prompts]
    policy = HFPolicy(
        model,
        temperature=0.8,
        max_new_tokens=128,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    reward = VisualBanditReward(root, out / "rollouts")
    set_seed(seed)
    scores = []
    for row in rows:
        images = []
        for relative in row["images"]:
            image = (root / relative).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError("Image escaped the dataset")
            images.append(str(image))
        for candidate in range(generations):
            completion = policy(row["prompt"], images)
            score = reward.score(row["task_id"], completion)
            scores.append(
                {
                    "task_id": row["task_id"],
                    "candidate": candidate,
                    "reward": score["reward"],
                    "execution_ok": score["execution_ok"],
                    "task_success": score["task_success"],
                    "completion": completion,
                }
            )
        jsonl(out / "scores.jsonl", scores)
    result = {
        "status": "complete",
        "model_identity": policy.identity,
        "dataset_sha256": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
        "readiness": grpo_readiness(scores),
        "prompts": len(rows),
        "generations": generations,
        "seed": seed,
        "contract": "contextual bandit: one sampled action plus fixed public finish",
        "generation": {"temperature": 0.8, "max_new_tokens": 128},
        "note": "Actual policy samples on TRAIN only; no RL optimizer update in calibration",
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompts", type=int, default=20)
    p.add_argument("--generations", type=int, default=4)
    args = p.parse_args()
    if args.prompts < 1 or args.generations < 2:
        p.error("Positive prompts and at least two generations required")
    print(
        json.dumps(
            calibrate(
                args.dataset, args.model, args.output, args.prompts, args.generations
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
