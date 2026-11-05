"""Serialized, equal-budget curriculum ablation; validation only, never TEST."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def validate_recipe(config):
    arms = config["arms"]
    if len(arms) != 3 or len({a["name"] for a in arms}) != 3:
        raise ValueError("Three distinct ablation arms required")
    totals = set()
    for arm in arms:
        if len(arm["datasets"]) != len(arm["budgets"]) or len(
            set(arm["datasets"])
        ) != len(arm["datasets"]):
            raise ValueError("Each unique source needs one explicit budget")
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in arm["budgets"]
        ):
            raise ValueError("Positive integer budgets required")
        totals.add(sum(arm["budgets"]))
    if len(totals) != 1 or config["steps"] < 1:
        raise ValueError("Ablation arms must have equal row/update budgets")


def select_arm(reports, evidence, retention_names, max_regression=0.05):
    eligible = [
        name
        for name in reports
        if all(
            evidence[name][metric]["success_at_1_delta"] >= -max_regression
            for metric in retention_names
        )
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda name: reports[name]["metrics"]["success_at_1"])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recipe", required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Use physical GPU 1 only")
    config = json.loads(Path(args.recipe).read_text())
    validate_recipe(config)
    out = Path(config["output"]).absolute()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Fresh experiment required")
    out.mkdir(parents=True)
    (out / "recipe.json").write_text(json.dumps(config, indent=2))
    files = list(Path("cad_env").rglob("*.py"))
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {"status": "waiting_data", "steps": {}, "arms": {}, "test_opened": False}

    def persist(**kw):
        state.update(kw)
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / "status.json")

    def check():
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in frozen.items()
        ):
            raise RuntimeError("Frozen curriculum source changed")

    def read(path):
        return json.loads(Path(path).read_text())

    def run(stage, module, *argv):
        check()
        persist(status=stage)
        command = [sys.executable, "-m", module, *map(str, argv)]
        with (out / (stage + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["steps"][stage] = {"status": "complete", "command": command}
        persist()

    def evaluate(stage, model, dataset, budget=8):
        target = out / stage
        run(
            stage,
            "cad_env.learning.workflow",
            "evaluate",
            "--model",
            model,
            "--dataset",
            dataset,
            "--output",
            target,
            "--disable-thinking",
            "--max-steps",
            budget,
            "--max-new-tokens",
            256,
            "--max-pixels",
            327680,
            "--max-context",
            8192,
        )
        result = read(target / "evaluation.json")
        if result["status"] != "complete" or result.get("limited_evaluation"):
            raise RuntimeError("Incomplete evaluation")
        return result

    try:
        persist()
        roots = sorted({d for a in config["arms"] for d in a["datasets"]})
        while not all((Path(d) / "summary.json").exists() for d in roots):
            check()
            time.sleep(10)
        from .train import check_training_source

        for root in roots:
            check_training_source(
                Path(root), Path(root) / "exports/train.feedback.jsonl"
            )
        persist(status="waiting_gpu")
        while True:
            path = Path(config["wait_for"])
            if path.exists():
                dependency = read(path)
                if dependency["status"] == "failed":
                    raise RuntimeError(
                        "Prior GPU experiment failed; investigate before continuing"
                    )
                if dependency["status"] == "complete":
                    break
            check()
            time.sleep(10)
        from .metrics import paired_comparison

        baseline = evaluate("baseline_units", config["parent"], config["validation"])
        for arm in config["arms"]:
            name = arm["name"]
            train, merged = out / (name + "-sft"), out / (name + "-merged")
            argv = [
                "--model",
                config["parent"],
                "--dataset",
                arm["datasets"][0],
                "--output",
                train,
                "--max-steps",
                config["steps"],
                "--save-steps",
                100,
                "--batch-size",
                1,
                "--gradient-accumulation",
                8,
                "--learning-rate",
                config["learning_rate"],
                "--numeric-loss-weight",
                8,
                "--max-length",
                8192,
                "--max-pixels",
                327680,
                "--max-images",
                4,
                "--vision-lora",
                "--source-row-budgets",
                *arm["budgets"],
            ]
            for root in arm["datasets"][1:]:
                argv += ["--extra-dataset", root]
            run(name + "_train", "cad_env.learning.train_modern", *argv)
            if read(train / "cad-training-run.json")["status"] != "complete":
                raise RuntimeError("Incomplete SFT")
            run(
                name + "_merge",
                "cad_env.learning.merge",
                "--adapter",
                train,
                "--output",
                merged,
            )
            candidate = evaluate(name + "_units", merged, config["validation"])
            evidence = {"vs_parent": paired_comparison(baseline, candidate)}
            for metric, dataset, reference, budget in config["retention"]:
                r = evaluate(name + "_" + metric, merged, dataset, budget)
                evidence[metric] = paired_comparison(read(reference), r)
            state["arms"][name] = evidence
            persist()
        reports = {
            a["name"]: read(out / (a["name"] + "_units") / "evaluation.json")
            for a in config["arms"]
        }
        names = [a["name"] for a in config["arms"]]
        pairwise = {
            names[i - 1] + "_to_" + names[i]: paired_comparison(
                reports[names[i - 1]], reports[names[i]]
            )
            for i in [1, 2]
        }
        selected = select_arm(
            reports, state["arms"], [r[0] for r in config["retention"]]
        )
        persist(
            selected_on_validation=selected,
            selection_rule="Maximum validation native success among arms with <=5pp retention regression; TEST sealed",
        )
        if selected is not None:
            run(
                "image_controls",
                "cad_env.learning.visual_controls",
                "--model",
                out / (selected + "-merged"),
                "--dataset",
                config["validation"],
                "--output",
                out / "image-controls",
                "--conditions",
                "no_goal",
                "blank_goal",
                "wrong_goal",
                "--max-steps",
                8,
            )
        persist(
            status="complete",
            pairwise=pairwise,
            interpretation="Equal-update/source-row ablation on validation only; no claim until paired intervals, retention and image controls are reviewed",
        )
    except Exception as exc:
        persist(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
