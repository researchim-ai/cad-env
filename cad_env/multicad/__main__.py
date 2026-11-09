"""CLI for optional multi-engine corpus generation, auditing and execution."""

import argparse
import json
from pathlib import Path

from .backend import MultiCADBackend
from .dataset import audit, generate
from .tasks import FAMILIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", required=True)
    gen.add_argument(
        "--engines",
        nargs="+",
        choices=["freecad", "cadquery"],
        default=["freecad", "cadquery"],
    )
    gen.add_argument("--seeds", type=int, default=3)
    gen.add_argument("--start-seed", type=int, default=0)
    gen.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    gen.add_argument(
        "--modes",
        nargs="+",
        choices=["create", "edit", "repair"],
        default=["create", "edit", "repair"],
    )
    gen.add_argument(
        "--modalities", nargs="+", choices=["text", "image"], default=["text", "image"]
    )
    gen.add_argument("--varied-edits", action="store_true")
    gen.add_argument("--python")

    gen.add_argument(
        "--view-mode", choices=["singleview", "multiview"], default="multiview"
    )
    check = sub.add_parser("audit")
    check.add_argument("--dataset", required=True)
    run = sub.add_parser("run")
    run.add_argument("--engine", required=True, choices=["freecad", "cadquery"])
    run.add_argument("--actions", required=True)
    run.add_argument("--reference")
    run.add_argument("--output", required=True)
    run.add_argument("--python")

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--engine", choices=["freecad", "cadquery"])
    evaluation.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    evaluation.add_argument("--final-test", action="store_true")
    evaluation.add_argument("--python")

    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "generate":
        result = generate(**args)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        if result["rejected"]:
            raise SystemExit(1)
    elif command == "audit":
        result = audit(args["dataset"])
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
    elif command == "evaluate":
        from cad_env.learning.policy import HFPolicy

        from .evaluate import evaluate

        policy = HFPolicy(
            args.pop("model"),
            max_new_tokens=256,
            max_context=8192,
            max_pixels=327680,
            chat_template_kwargs={"enable_thinking": False},
        )
        result = evaluate(policy=policy, **args)
        print(json.dumps(result["metrics"], indent=2))
    else:
        actions = json.loads(Path(args.pop("actions")).read_text())
        reference = args.pop("reference")
        output = args.pop("output")
        backend = MultiCADBackend(**args)
        result = backend.run(
            actions,
            output,
            reference_actions=json.loads(Path(reference).read_text())
            if reference
            else None,
        )
        print(json.dumps(result, indent=2))
        if not result["execution_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
