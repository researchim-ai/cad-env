"""Concurrent native validation using one GPU policy and an explicit batch profile."""

import argparse
import json
from pathlib import Path

from .batch_policy import BatchedHFPolicy, PolicyBatcher
from .parallel_protocol import evaluate_policy


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ["model", "dataset", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    policy = BatchedHFPolicy(
        args.model,
        temperature=0,
        max_new_tokens=256,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    batcher = PolicyBatcher(policy, args.batch_size)
    try:
        report = evaluate_policy(
            args.dataset,
            lambda seed: batcher,
            args.output,
            workers=args.workers,
            max_steps=args.max_steps,
            limit=args.limit,
            model_identity=policy.identity,
            policy_protocol={
                "max_new_tokens": 256,
                "temperature": 0,
                "max_context": 8192,
                "max_pixels": 327680,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(json.dumps(report["metrics"], indent=2))
    finally:
        batcher.close()
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "batch-execution.json").write_text(
            json.dumps(
                {
                    "batch_size_limit": args.batch_size,
                    "episode_workers": args.workers,
                    "stats": batcher.stats,
                    "latency_comparable_to_serial": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
