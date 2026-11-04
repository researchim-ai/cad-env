"""Evaluate image interventions with the unchanged native controller.

Memoization is restricted to greedy inference and identical public text/image
content. Every episode is still executed and independently graded in FreeCAD.
Cached runs are unsuitable for comparing inference latency.
"""

import argparse
import hashlib
import json
from pathlib import Path


def uniform_goal_size(dataset, split):
    """Read only public image dimensions for the selected control cases."""
    from PIL import Image

    root = Path(dataset).resolve()
    sizes = set()
    for line in (root / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if not entry["accepted"] or entry["split"] != split:
            continue
        task = json.loads(
            (root / "episodes" / entry["task_id"] / "task.json").read_text()
        )
        for relative in task.get("goal_images", []):
            image = (root / relative).resolve()
            if not image.is_relative_to(root):
                raise ValueError("Goal image escaped the dataset")
            with Image.open(image) as picture:
                sizes.add(picture.size)
    if len(sizes) != 1:
        raise ValueError(
            "Shape-preserving blank control requires uniform goal image dimensions"
        )
    return next(iter(sizes))


class GreedyPublicCache:
    def __init__(self, policy):
        if policy.temperature != 0:
            raise ValueError("Memoization requires deterministic greedy decoding")
        self.policy = policy
        self.responses = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, messages, image_paths=()):
        key = hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
        )
        for path in image_paths:
            key.update(hashlib.sha256(Path(path).read_bytes()).digest())
        digest = key.hexdigest()
        if digest in self.responses:
            self.hits += 1
            return self.responses[digest]
        response = self.policy(messages, image_paths)
        self.responses[digest] = response
        self.misses += 1
        return response


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["no_goal", "blank_goal", "wrong_goal"],
        choices=["full", "no_goal", "blank_goal", "wrong_goal"],
    )
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()
    from transformers import set_seed

    from .policy import HFPolicy
    from .protocol import evaluate_policy

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty control output directory")
    out.mkdir(parents=True, exist_ok=True)
    blank_size = (
        uniform_goal_size(args.dataset, args.split)
        if "blank_goal" in args.conditions
        else None
    )
    policy = HFPolicy(
        args.model,
        max_new_tokens=256,
        temperature=0,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    cached = GreedyPublicCache(policy)

    def factory(seed):
        set_seed(seed)
        if condition == "blank_goal":
            # The frozen evaluator creates its default blank after factory().
            # Supply a matching-size blank first, preserving the image grid and
            # token count as well as the public text and model weights.
            from PIL import Image

            blank = out / condition / "blank-goal.png"
            if not blank.exists():
                Image.new("RGB", blank_size, "white").save(blank)
        return cached

    summary = {
        "status": "running",
        "conditions": {},
        "memoization": "Greedy public text plus image content only; native grading uncached; latency not comparable",
        "blank_goal_size": blank_size,
    }

    def persist():
        (out / "controls.json").write_text(json.dumps(summary, indent=2))

    persist()
    try:
        for condition in args.conditions:
            report = evaluate_policy(
                args.dataset,
                factory,
                out / condition,
                split=args.split,
                final_test=args.final_test,
                max_steps=args.max_steps,
                model_identity=policy.identity,
                image_condition=condition,
                policy_protocol={
                    "max_new_tokens": 256,
                    "temperature": 0.0,
                    "max_context": 8192,
                    "max_pixels": 327680,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            summary["conditions"][condition] = report["metrics"]
            summary.update(cache_hits=cached.hits, cache_misses=cached.misses)
            persist()
        summary["status"] = "complete"
        persist()
    except Exception as error:
        summary.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
