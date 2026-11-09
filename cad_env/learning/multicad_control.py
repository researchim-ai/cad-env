"""Remove exactly the two goal images, preserving text and current-state input."""

import argparse
import copy
import json
from pathlib import Path


class NoGoalPolicy:
    def __init__(self, policy):
        from .visual_controls import GreedyPublicCache

        self.cached = GreedyPublicCache(policy)
        self.base = policy

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, messages, image_paths=()):
        if len(image_paths) != 3:
            raise ValueError("Control requires specification, goal view, current view")
        transformed = copy.deepcopy(messages)
        index = 0
        for message in transformed:
            if isinstance(message["content"], list):
                blocks = []
                for block in message["content"]:
                    if block.get("type") == "image":
                        index += 1
                        if index <= 2:
                            continue
                    blocks.append(block)
                message["content"] = blocks
        if index != 3:
            raise ValueError("Image placeholders do not match the control")
        return self.cached(transformed, image_paths[2:])


def main():
    p = argparse.ArgumentParser(__doc__)
    for arg in ["dataset", "model", "output", "engine", "python"]:
        p.add_argument("--" + arg, required=arg not in ["python"])
    args = vars(p.parse_args())
    from cad_env.multicad.evaluate import evaluate

    from .policy import HFPolicy

    model = args.pop("model")
    policy = NoGoalPolicy(
        HFPolicy(
            model,
            max_new_tokens=256,
            max_context=8192,
            max_pixels=327680,
            chat_template_kwargs={"enable_thinking": False},
        )
    )
    report = evaluate(policy=policy, **args)
    report["image_condition"] = "no_goal"
    report["control_note"] = (
        "Identical public text, model and decoding; two goal images removed; current image retained. Public greedy memoization, native grading uncached."
    )
    (Path(args["output"]) / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
