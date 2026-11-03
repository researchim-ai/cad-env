"""Task-level evaluation and paired cluster-bootstrap comparisons."""

import math
import random
import statistics
from collections import defaultdict


def summarize_episodes(episodes):
    if not episodes:
        raise ValueError("Evaluation has no episodes")
    groups = defaultdict(list)
    for row in episodes:
        groups[row["task_id"]].append(row)
    first = []
    for task_id, attempts in groups.items():
        if len({r["attempt"] for r in attempts}) != len(attempts):
            raise ValueError("Duplicate attempt for task " + task_id)
        first.append(min(attempts, key=lambda r: r["attempt"]))
    success = lambda rows: sum(bool(r["task_success"]) for r in rows) / len(rows)
    breakdown = {}
    for dimension in ("family", "mode"):
        breakdown[dimension] = {
            value: {"tasks": len(selected), "success_at_1": success(selected)}
            for value in sorted({r[dimension] for r in first})
            if (selected := [r for r in first if r[dimension] == value])
        }
    total_steps = sum(r.get("steps", 0) for r in episodes)
    return {
        "tasks": len(groups),
        "episodes": len(episodes),
        "success_at_1": success(first),
        "success_at_budget": sum(
            any(r["task_success"] for r in rows) for rows in groups.values()
        )
        / len(groups),
        "attempts_per_task": sorted({len(rows) for rows in groups.values()}),
        "mean_reward": statistics.mean(r["reward"] for r in episodes),
        "invalid_action_rate": sum(r.get("invalid_actions", 0) for r in episodes)
        / max(total_steps, 1),
        "mean_steps": statistics.mean(r.get("steps", 0) for r in episodes),
        "median_seconds": statistics.median(r["elapsed_seconds"] for r in episodes),
        "truncation_rate": sum(r.get("truncated", False) for r in episodes)
        / len(episodes),
        "breakdown": breakdown,
    }


def paired_comparison(baseline, candidate, *, seed=42, resamples=2000):
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("Only complete evaluations can be compared")
    if baseline.get("limited_evaluation") or candidate.get("limited_evaluation"):
        raise ValueError("Limited debug evaluations cannot promote checkpoints")
    for key in ("dataset_digest", "protocol_digest", "task_ids"):
        if baseline[key] != candidate[key]:
            raise ValueError("Incomparable evaluations: " + key)

    def first_rows(report):
        return {r["task_id"]: r for r in report["episodes"] if r["attempt"] == 0}

    before, after = first_rows(baseline), first_rows(candidate)
    if set(before) != set(after) or sorted(before) != sorted(baseline["task_ids"]):
        raise ValueError("Missing tasks or incomplete evaluation")
    clusters = defaultdict(list)
    for task_id in before:
        if before[task_id]["split_group"] != after[task_id]["split_group"]:
            raise ValueError("Lineage mismatch")
        clusters[before[task_id]["split_group"]].append(
            float(after[task_id]["task_success"])
            - float(before[task_id]["task_success"])
        )
    values = list(clusters.values())
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sampled = [rng.choice(values) for _ in values]
        flattened = [d for cluster in sampled for d in cluster]
        deltas.append(statistics.mean(flattened))
    deltas.sort()
    low, high = (
        deltas[int(0.025 * resamples)],
        deltas[min(resamples - 1, math.ceil(0.975 * resamples))],
    )
    observed = statistics.mean(d for cluster in values for d in cluster)
    return {
        "paired_task_count": len(before),
        "independent_lineage_groups": len(values),
        "success_at_1_delta": observed,
        "cluster_bootstrap_95_ci": [low, high],
        "seed": seed,
        "resamples": resamples,
        "improvement_supported": low > 0 and len(values) >= 20,
        "note": "Resample parent/base-design groups, not correlated edit/create/image rows. This is within-benchmark evidence, not proof of general CAD competence.",
    }
