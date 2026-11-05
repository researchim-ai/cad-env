"""Sequential GPU-1 mixed-SFT experiment, with native before/after evaluation.

Invoke using the pinned Qwen environment; geometry runs in a separate Python.
Long stages persist state and never promote partial evaluations.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recipe", required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("This measured recipe requires physical GPU 1")
    recipe_path = Path(args.recipe).resolve()
    config = json.loads(recipe_path.read_text())
    root = Path.cwd()
    out = Path(config["output"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh experiment output")
    out.mkdir(parents=True)
    shutil.copyfile(recipe_path, out / "recipe.json")
    sources = [
        *Path("cad_env/agent").glob("*.py"),
        *Path("cad_env/multicad").glob("*.py"),
    ]
    sources += [
        Path("cad_env/learning") / name
        for name in [
            "train_modern.py",
            "train.py",
            "mixing.py",
            "feedback_normalization.py",
            "numeric_loss.py",
            "merge.py",
            "guards.py",
            "checkpoints.py",
            "policy.py",
            "multicad_comparison.py",
            "multicad_control.py",
            "multicad_experiment.py",
        ]
    ]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out / "source-freeze.json").write_text(json.dumps(frozen, indent=2))
    state = {
        "status": "waiting_datasets",
        "steps": {},
        "config_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "optimizer_run": False,
    }

    def persist(**kw):
        state.update(kw)
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(out / "status.json")

    def check():
        for name, digest in frozen.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Experiment source drift: " + name)

    def run(stage, module, *argv, python=None):
        check()
        persist(status=stage)
        command = [python or sys.executable, "-m", module, *map(str, argv)]
        with (out / (stage + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["steps"][stage] = {"status": "complete", "command": command}
        persist()

    def read(path):
        return json.loads(Path(path).read_text())

    def evaluate(stage, model, dataset, output, engine=None, final=False):
        argv = [
            "--dataset",
            dataset,
            "--model",
            model,
            "--output",
            output,
            "--python",
            str((root / config["geometry_python"]).absolute()),
            "--blender",
            str((root / config["blender"]).resolve()),
        ]
        if engine:
            argv += ["--engine", engine]
        if final:
            argv += ["--split", "test", "--final-test"]
        run(stage, "cad_env.multicad", "evaluate", *argv)
        report = read(Path(output) / "evaluation.json")
        if report["status"] != "complete":
            raise RuntimeError("Incomplete native evaluation")
        return report

    try:
        persist()
        while not all(
            (Path(d) / "summary.json").exists() for d in config["datasets"].values()
        ):
            check()
            time.sleep(10)
        from scipy.spatial import cKDTree

        from cad_env.multicad.dataset import audit

        # Fail before loading a VLM if the evaluation dependency is absent.
        cKDTree([[0.0, 0.0, 0.0]])

        subprocess.run(
            [
                str((root / config["geometry_python"]).absolute()),
                "-c",
                "import sys, cadquery; print(sys.prefix)",
            ],
            check=True,
        )

        datasets = {}
        for engine, directory in config["datasets"].items():
            report = audit(directory)
            summary = read(Path(directory) / "summary.json")
            if not report["ok"] or summary["rejected"]:
                raise RuntimeError("Native corpus audit failed: " + directory)
            manifest = [
                json.loads(line)
                for line in (Path(directory) / "manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            groups = {
                s: len(
                    {
                        r["split_group"]
                        for r in manifest
                        if r["accepted"] and r["split"] == s
                    }
                )
                for s in ["train", "validation", "test"]
            }
            if (
                groups["validation"] < config["gates"]["min_validation_groups"]
                or groups["test"] < config["gates"]["min_test_groups"]
            ):
                raise RuntimeError("Insufficient independent holdout geometries")
            datasets[engine] = {
                "digest": hashlib.sha256(
                    (Path(directory) / "summary.json").read_bytes()
                ).hexdigest(),
                "groups": groups,
                "accepted": summary["accepted"],
            }
        persist(status="waiting_gpu_queue", datasets=datasets)
        while True:
            dependency = Path(config["wait_for"])
            if dependency.exists():
                previous = read(dependency)
                if previous["status"] in ["complete", "failed"]:
                    persist(prior_grpo_status=previous)
                    break
            check()
            time.sleep(10)
        baseline = {}
        for engine, dataset in config["datasets"].items():
            baseline[engine] = evaluate(
                "baseline_" + engine,
                config["parent"],
                dataset,
                out / ("baseline-" + engine),
                engine,
            )
        train = out / "sft"
        merged = out / "merged"
        training = config["training"]
        data = list(config["datasets"].values())
        argv = [
            "--model",
            config["parent"],
            "--dataset",
            data[0],
            "--output",
            train,
            "--rows-per-source",
            training["rows_per_source"],
            "--max-steps",
            training["steps"],
            "--save-steps",
            100,
            "--batch-size",
            training["batch_size"],
            "--gradient-accumulation",
            training["gradient_accumulation"],
            "--learning-rate",
            training["learning_rate"],
            "--numeric-loss-weight",
            training["numeric_loss_weight"],
            "--max-length",
            8192,
            "--max-pixels",
            327680,
            "--max-images",
            4,
            "--vision-lora",
        ]
        for directory in data[1:] + config["replay"]:
            argv += ["--extra-dataset", directory]
        run("mixed_sft", "cad_env.learning.train_modern", *argv)
        training_report = read(train / "cad-training-run.json")
        if training_report["status"] != "complete":
            raise RuntimeError("SFT did not complete")
        persist(
            optimizer_run=True, training_report=str(train / "cad-training-run.json")
        )
        run("merge", "cad_env.learning.merge", "--adapter", train, "--output", merged)
        candidate = {}
        comparisons = {}
        from .multicad_comparison import combined, compare

        for engine, dataset in config["datasets"].items():
            candidate[engine] = evaluate(
                "candidate_" + engine,
                merged,
                dataset,
                out / ("candidate-" + engine),
                engine,
            )
            comparisons[engine] = compare(baseline[engine], candidate[engine])
        joint = combined([(baseline[e], candidate[e]) for e in baseline])
        persist(comparisons=comparisons, joint_comparison=joint)
        retained = {}
        for name, directory, baseline_file, steps in [
            ("retention", "gpu1-corpus-v2", "vision-leap-c-retention", 16),
            ("grounding", "visual-grounding-v1", "vision-leap-c-grounding", 8),
            ("complex", "complex-cad-v3", "vision-leap-c-complex", 16),
            (
                "entities",
                "visual-grounding-v4-entities",
                "vision-leap-entities-candidate",
                8,
            ),
        ]:
            target = out / ("retention-" + name)
            run(
                "retention_" + name,
                "cad_env.learning.workflow",
                "evaluate",
                "--model",
                merged,
                "--dataset",
                "artifacts/" + directory,
                "--output",
                target,
                "--disable-thinking",
                "--max-steps",
                steps,
                "--max-new-tokens",
                256,
                "--max-pixels",
                327680,
                "--max-context",
                8192,
            )
            retained[name] = read(target / "evaluation.json")["metrics"]["success_at_1"]
            run(
                "retention_compare_" + name,
                "cad_env.learning.workflow",
                "compare",
                "--baseline",
                "artifacts/" + baseline_file + "/evaluation.json",
                "--candidate",
                target / "evaluation.json",
                "--output",
                out / ("comparison-retention-" + name + ".json"),
            )
        image_effects = {}
        from .multicad_comparison import image_effect

        for engine, dataset in config["datasets"].items():
            target = out / ("no-goal-" + engine)
            run(
                "no_goal_" + engine,
                "cad_env.learning.multicad_control",
                "--dataset",
                dataset,
                "--model",
                merged,
                "--output",
                target,
                "--engine",
                engine,
                "--python",
                (root / config["geometry_python"]).absolute(),
                "--blender",
                (root / config["blender"]).resolve(),
            )
            image_effects[engine] = image_effect(
                candidate[engine], read(target / "evaluation.json")
            )
        persist(image_effects=image_effects)
        g = config["gates"]
        eligible = (
            all(
                r["modes"].get("create", {}).get("success_at_1_delta", 0) >= 0.4
                and r["modes"].get("create", {}).get("improvement_supported", False)
                for r in image_effects.values()
            )
            and joint["improvement_supported"]
            and joint["success_at_1_delta"] >= g["min_gain"]
            and retained["retention"] >= g["min_retention"]
            and retained["grounding"] >= 0.25
            and retained["complex"] >= 0.79
            and retained["entities"] >= 0.81
            and all(
                r["after_success"] >= g["min_engine_success"]
                and r["success_at_1_delta"] >= -g["max_engine_regression"]
                for r in comparisons.values()
            )
        )
        persist(retention=retained, validation_gates_passed=eligible)
        if eligible:
            for engine, dataset in config["datasets"].items():
                before = evaluate(
                    "final_parent_" + engine,
                    config["parent"],
                    dataset,
                    out / ("final-parent-" + engine),
                    engine,
                    True,
                )
                after = evaluate(
                    "final_candidate_" + engine,
                    merged,
                    dataset,
                    out / ("final-candidate-" + engine),
                    engine,
                    True,
                )
                (out / ("comparison-final-" + engine + ".json")).write_text(
                    json.dumps(compare(before, after), indent=2)
                )
        persist(
            status="complete",
            interpretation="Report validation and retention whether or not promotion gates pass; no broad CAD mastery claim",
        )
    except Exception as exc:
        persist(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
