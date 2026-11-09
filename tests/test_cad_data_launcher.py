import argparse
import json

import pytest

from scripts.generate_cad_data import commands, merge_indices


def test_all_suite_commands_preserve_counts_split_and_exclusions(tmp_path):
    args = argparse.Namespace(
        output=str(tmp_path),
        suite="all",
        per_family_level=7,
        levels=["easy", "hard"],
        split="validation",
        seed=20,
        jobs=2,
        exclude_pairs=["heldout/pairs.jsonl"],
        plan_only=True,
    )
    jobs = dict(commands(args))
    assert set(jobs) == {"industrial", "curriculum"}
    for cmd in jobs.values():
        assert cmd[cmd.index("--split") + 1] == "validation"
        assert cmd[cmd.index("--exclude-pairs") + 1] == "heldout/pairs.jsonl"
        assert "--plan-only" in cmd
    cmd = jobs["curriculum"]
    assert cmd[cmd.index("--per-level") + 1] == "21"
    cmd = jobs["industrial"]
    assert cmd[cmd.index("--per-family-level") + 1] == "7"


def fixture(root, name, group):
    folder = root / name / "paired"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"status": "complete"}))
    row = {
        "task_id": group,
        "split_group": group,
        "family": name,
        "difficulty": "easy",
        "representations": {
            "iso": {"dataset": f"/{name}/iso"},
            "gost": {"dataset": f"/{name}/gost"},
        },
    }
    (folder / "pairs.jsonl").write_text(json.dumps(row) + "\n")


def test_merged_index_retains_both_profiles_and_training_roots(tmp_path):
    fixture(tmp_path, "industrial", "a")
    fixture(tmp_path, "curriculum", "b")
    pairs = merge_indices(tmp_path, ["industrial", "curriculum"])
    assert len(pairs) == 2
    assert (
        len(json.loads((tmp_path / "dataset-sources.json").read_text())["datasets"])
        == 4
    )
    summary = json.loads((tmp_path / "paired/summary.json").read_text())
    assert summary["documents"] == 4


def test_duplicate_lineage_is_not_silently_pooled(tmp_path):
    fixture(tmp_path, "industrial", "same")
    fixture(tmp_path, "curriculum", "same")
    with pytest.raises(ValueError, match="Duplicate physical"):
        merge_indices(tmp_path, ["industrial", "curriculum"])
