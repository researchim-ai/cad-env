"""Native evaluation of the explicitly separate complete-plan agent contract."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ["model", "dataset", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument(
        "--image-condition",
        choices=["full", "no_goal", "blank_goal", "wrong_goal"],
        default="full",
    )
    a = p.parse_args()
    from .batch_policy import BatchedHFPolicy, PolicyBatcher
    from .parallel_protocol import evaluate_policy
    from .perception_plan import PlannedPolicy, PublicCropper

    model = BatchedHFPolicy(
        a.model,
        temperature=0,
        max_new_tokens=512,
        max_context=8192,
        max_pixels=327680,
        chat_template_kwargs={"enable_thinking": False},
    )
    batcher = PolicyBatcher(model, 4)
    out = Path(a.output)
    cropper = PublicCropper(out / "public-crops")
    try:
        result = evaluate_policy(
            a.dataset,
            lambda seed: PlannedPolicy(batcher, cropper, out / "plans", seed),
            out,
            workers=8,
            max_steps=a.max_steps,
            image_condition=a.image_condition,
            model_identity=model.identity,
            policy_protocol={
                "action_contract": "perception_plan_v1",
                "executor": "deterministic typed-tool plan compiler",
                "perception_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("perception_plan.py").read_bytes()
                ).hexdigest(),
                "max_new_tokens": 512,
                "temperature": 0,
                "max_context": 8192,
                "max_pixels": 327680,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(json.dumps(result["metrics"], indent=2))
    finally:
        batcher.close()


if __name__ == "__main__":
    main()
