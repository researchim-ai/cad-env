import argparse
import json
from pathlib import Path

from cad_env.agent import FAMILIES, MODES, FreeCADBackend

from .factory import generate_dataset, jsonl
from .rewards import CADReward, group_statistics


def main():
    parser = argparse.ArgumentParser(
        description="Verified FreeCAD SFT/GRPO data factory"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--freecad")
    generate = commands.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--seeds", type=int, default=10)
    generate.add_argument("--start-seed", type=int, default=0)
    generate.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    generate.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    generate.add_argument("--holdout-families", nargs="*", choices=FAMILIES, default=[])
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--timeout", type=float, default=30)
    generate.add_argument("--freecad")
    generate.add_argument("--no-render", action="store_true")
    generate.add_argument("--preferences", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--dataset", required=True)
    score.add_argument("--completions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--freecad")
    score.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit_dataset

        report = audit_dataset(args.dataset)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "doctor":
        import tempfile

        from cad_env.agent import make_task
        from cad_env.agent.tasks import expected_metrics

        task = make_task()
        with tempfile.TemporaryDirectory(prefix="cad-doctor-") as path:
            result = FreeCADBackend(args.freecad).run(
                task["oracle_actions"],
                path,
                reference_actions=task["reference_actions"],
                expected_metrics=expected_metrics(task),
            )
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("execution_ok", "runtime", "validation", "error")
                },
                indent=2,
            )
        )
        return 0 if result.get("validation", {}).get("task_success") else 1
    if args.command == "generate":
        summary = generate_dataset(
            args.output,
            seeds=args.seeds,
            start_seed=args.start_seed,
            families=args.families,
            modes=args.modes,
            executable=args.freecad,
            timeout=args.timeout,
            workers=args.workers,
            render=not args.no_render,
            preferences=args.preferences,
            holdout_families=args.holdout_families,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["rejected"] == 0 else 1
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Score output must be a new or empty directory")
    reward = CADReward(
        Path(args.dataset) / "episodes",
        output / "rollouts",
        executable=args.freecad,
        timeout=args.timeout,
    )
    rows = [
        json.loads(line)
        for line in Path(args.completions).read_text().splitlines()
        if line.strip()
    ]
    scores = []
    groups = {}
    for row in rows:
        score = reward.score(row["task_id"], row["completion"])
        record = {**row, **score}
        scores.append(record)
        groups.setdefault(row["task_id"], []).append(score["reward"])
    jsonl(output / "scores.jsonl", scores)
    report = {k: group_statistics(v) for k, v in groups.items() if len(v) >= 2}
    (output / "groups.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"scored": len(scores), "groups": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
