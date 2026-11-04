"""Publication-style plot from complete native evaluations, including failures."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def rate_interval(rows, seed=42, resamples=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split_group"]].append(float(row["task_success"]))
    values = list(groups.values())
    rng = random.Random(seed)
    rates = sorted(
        statistics.mean(v for _ in values for v in rng.choice(values))
        for _ in range(resamples)
    )
    rate = statistics.mean(float(r["task_success"]) for r in rows)
    return rate, rates[int(0.025 * resamples)], rates[int(0.975 * resamples)]


def plot(root, candidate, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root).resolve()
    reports = {}
    for stage in ("baseline", candidate):
        reports[stage] = {}
        for name in ("grounding", "complex", "retention"):
            path = root / f"vision-leap-{stage}-{name}" / "evaluation.json"
            report = json.loads(path.read_text())
            if report["status"] != "complete" or report.get("limited_evaluation"):
                raise ValueError("Only complete native evaluations may enter the plot")
            if report["protocol"]["attempts"] != 1:
                raise ValueError("This figure reports one attempt per task")
            reports[stage][name] = report
    for name in ("grounding", "complex", "retention"):
        a, b = reports["baseline"][name], reports[candidate][name]
        for key in ("dataset_digest", "protocol_digest", "task_ids"):
            if a[key] != b[key]:
                raise ValueError("Unmatched plot inputs: " + key)
    modalities = {}
    for task_id in reports["baseline"]["complex"]["task_ids"]:
        task = json.loads(
            (root / "complex-cad-v3/episodes" / task_id / "task.json").read_text()
        )
        modalities[task_id] = task["input_modality"]
    labels = [
        "Выбор отверстий\nбез номеров",
        "Построение\nпо чертежу",
        "Построение\nпо тексту",
        "Прежние\nзадачи",
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.4), layout="constrained")
    colors = ["#8193a5", "#1268a0"]
    values = {}
    for offset, stage in enumerate(("baseline", candidate)):
        r = reports[stage]
        slices = [
            r["grounding"]["episodes"],
            [
                e
                for e in r["complex"]["episodes"]
                if modalities[e["task_id"]] == "image"
            ],
            [e for e in r["complex"]["episodes"] if modalities[e["task_id"]] == "text"],
            r["retention"]["episodes"],
        ]
        stats = [rate_interval(rows) for rows in slices]
        heights = np.array([s[0] for s in stats]) * 100
        errors = (
            np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]) * 100
        )
        x = np.arange(4) + (-0.19 if offset == 0 else 0.19)
        ax.bar(
            x,
            heights,
            width=0.34,
            color=colors[offset],
            yerr=errors,
            capsize=3,
            error_kw={"elinewidth": 1},
            label="Исходный checkpoint (SFT200)"
            if offset == 0
            else f"После этапа {candidate.upper()}",
        )
        for pos, stats_row, rows in zip(x, stats, slices):
            count = sum(e["task_success"] for e in rows)
            ax.text(
                pos,
                stats_row[2] * 100 + 2.2,
                f"{count}/{len(rows)}",
                ha="center",
                fontsize=9,
            )
        values[stage] = [
            {
                "successes": sum(e["task_success"] for e in rows),
                "tasks": len(rows),
                "success": s[0],
                "cluster_bootstrap_95_ci": [s[1], s[2]],
            }
            for rows, s in zip(slices, stats)
        ]
    ax.set(
        xticks=np.arange(4),
        xticklabels=labels,
        ylim=(0, 112),
        ylabel="Нативный успех, %",
        title="Qwen3.5-4B в FreeCAD — validation",
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01))
    fig.text(
        0.5,
        -0.025,
        "Greedy, одна попытка. 95% bootstrap по исходным деталям. Внутренние процедурные наборы.",
        ha="center",
        fontsize=9,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps({"labels": labels, "values": values}, indent=2, ensure_ascii=False)
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--candidate", choices=["b", "c"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plot(args.artifacts, args.candidate, args.output)


if __name__ == "__main__":
    main()
