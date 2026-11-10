"""Deterministic coverage sampling of paired TRAIN decisions, without replacement."""

import random
from collections import Counter, defaultdict

QUOTAS = {
    "cb_sketch": 0.20,
    "cb_pocket": 0.20,
    "finish": 0.10,
    "late": 0.20,
    "recovery": 0.05,
}


def category(row):
    if row.get("recovery"):
        return "recovery"
    op = row["operation"]
    if row["counterbore"] and op == "sketch_circles":
        return "cb_sketch"
    if row["counterbore"] and op == "pocket":
        return "cb_pocket"
    if op == "finish":
        return "finish"
    if row["progress"] >= 0.65:
        return "late"
    return "other"


def select_decisions(decisions, count, seed=20261110):
    """Round-robin parents within quotas; report shortages rather than duplicate rows.

    Categories are exclusive. Recovery is reported separately, never fabricated
    from clean teacher prefixes. Remaining slots sample the unused pool.
    """
    if count <= 0 or count > len(decisions):
        raise ValueError("Positive budget within available unique decisions required")
    keys = [
        (r["split_group"], r["step_id"], r.get("recovery", False)) for r in decisions
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate causal decision")
    if any(r["split"] != "train" for r in decisions):
        raise ValueError("TRAIN decisions only")
    rng = random.Random(seed)
    pools = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(decisions):
        pools[category(row)][row["split_group"]].append(i)

    def balanced(groups):
        parents = sorted(groups)
        rng.shuffle(parents)
        for parent in parents:
            rng.shuffle(groups[parent])
        while parents:
            remaining = []
            for parent in parents:
                yield groups[parent].pop()
                if groups[parent]:
                    remaining.append(parent)
            parents = remaining

    chosen, shortages = [], {}
    for cat, fraction in QUOTAS.items():
        want = int(count * fraction)
        available = list(balanced(pools[cat]))
        chosen.extend(available[:want])
        shortages[cat] = max(0, want - len(available))
    used = set(chosen)
    rest = defaultdict(list)
    for i, row in enumerate(decisions):
        if i not in used:
            rest[row["split_group"]].append(i)
    chosen.extend(list(balanced(rest))[: count - len(chosen)])
    rng.shuffle(chosen)
    selected = [decisions[i] for i in chosen]
    return selected, {
        "decisions": len(selected),
        "unique_parents": len({r["split_group"] for r in selected}),
        "categories": dict(Counter(category(r) for r in selected)),
        "quota_shortages": shortages,
        "recovery_available": sum(category(r) == "recovery" for r in decisions),
        "scope": "Clean-prefix coverage is not autonomous recovery training or evidence of model improvement.",
    }
