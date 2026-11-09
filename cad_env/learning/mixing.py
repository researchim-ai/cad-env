"""Explicit deterministic source budgets; no implicit dataset-root duplication."""

import random


def mix_sources(datasets, rows_per_source=None, seed=42, source_seeds=None):
    from datasets import concatenate_datasets

    if not datasets or any(len(d) == 0 for d in datasets):
        raise ValueError("Every source must be nonempty")
    if source_seeds is not None and (
        len(source_seeds) != len(datasets)
        or any(type(s) is not int for s in source_seeds)
    ):
        raise ValueError("One integer sampling seed per source required")
    if rows_per_source is None:
        return concatenate_datasets(datasets)
    budgets = (
        rows_per_source
        if isinstance(rows_per_source, (list, tuple))
        else [rows_per_source] * len(datasets)
    )
    if len(budgets) != len(datasets) or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in budgets
    ):
        raise ValueError("Positive source budget required")
    selected = []
    for index, dataset in enumerate(datasets):
        rng = random.Random(
            source_seeds[index] if source_seeds is not None else seed + index
        )
        indices = []
        budget = budgets[index]
        while len(indices) < budget:
            cycle = list(range(len(dataset)))
            rng.shuffle(cycle)
            indices.extend(cycle)
        selected.append(dataset.select(indices[:budget]))
    return concatenate_datasets(selected)
