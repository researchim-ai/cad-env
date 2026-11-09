import json
from collections import Counter

from scripts.benchmark_paired_drawings import select_pairs
from scripts.generate_curriculum_drawings import select_plan


def tasks(plan):
    return [t for shard in plan["shards"] for t in shard["tasks"]]


def test_three_levels_balanced_and_naturally_split():
    plan = select_plan()
    rows = tasks(plan)
    assert len(rows) == len({t["split_group"] for t in rows}) == 90
    assert Counter(t["difficulty"] for t in rows) == {
        "easy": 30,
        "medium": 30,
        "hard": 30,
    }
    assert {t["split"] for t in rows} == {"validation"}
    assert all(75 <= int(t["split_group"][:8], 16) % 100 < 90 for t in rows)
    for family in {t["family"] for t in rows}:
        ranges = {
            level: {
                len(t["oracle_actions"])
                for t in rows
                if t["family"] == family and t["difficulty"] == level
            }
            for level in ("easy", "medium", "hard")
        }
        assert max(ranges["easy"]) < min(ranges["medium"])
        assert max(ranges["medium"]) < min(ranges["hard"])


def test_train_validation_disjoint_and_exclusions():
    validation = tasks(select_plan(per_level=3))
    excluded = {t["split_group"] for t in validation}
    train = tasks(select_plan(per_level=3, split="train"))
    assert not excluded & {t["split_group"] for t in train}
    assert all(int(t["split_group"][:8], 16) % 100 < 75 for t in train)
    fresh = tasks(select_plan(per_level=3, excluded=excluded))
    assert not excluded & {t["split_group"] for t in fresh}


def test_arbitrary_positive_counts_and_determinism():
    assert select_plan(per_level=7) == select_plan(per_level=7)
    assert len(tasks(select_plan(per_level=7))) == 21
    assert len(tasks(select_plan(per_level=1, levels=("easy",)))) == 1


def test_baseline_preserves_all_90_tasks_and_levels(tmp_path):
    rows = tasks(select_plan())
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(t) + "\n" for t in rows))
    selected = select_pairs(tmp_path, count=90)
    assert {t["task_id"] for t in selected} == {t["task_id"] for t in rows}
    assert Counter((t["family"], t["difficulty"]) for t in selected) == {
        (family, level): 10
        for family in {t["family"] for t in rows}
        for level in ("easy", "medium", "hard")
    }
