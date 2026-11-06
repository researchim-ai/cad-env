import json
from collections import Counter
from pathlib import Path

import pytest

from cad_env.learning.visual_followup import training_command
from cad_env.learning.visual_holdout import validation_tasks


def test_followup_is_one_epoch_and_keeps_checkpoints():
    config = json.loads(Path("recipes/visual-followup-500-qwen35.json").read_text())
    command = training_command(config)
    assert command[command.index("--max-steps") + 1] == "500"
    assert command[command.index("--save-total-limit") + 1] == "6"
    assert command[command.index("--lr-scheduler-type") + 1] == "constant_with_warmup"
    config["sources"][0]["rows"] -= 1
    with pytest.raises(ValueError):
        training_command(config)


def test_fresh_validation_has_balanced_independent_parents_without_training_rows():
    tasks = validation_tasks()
    assert len(tasks) == 640
    assert {t["split"] for t in tasks} == {"validation"}
    parents = {}
    for task in tasks:
        parents.setdefault(task["split_group"], task)
    assert len(parents) == 40
    assert Counter(len(t["oracle_actions"]) - 1 for t in parents.values()) == {
        1: 10,
        2: 10,
        3: 10,
        4: 10,
    }
    # Check against every actual training-source lineage, not merely task IDs.
    config = json.loads(Path("recipes/visual-followup-500-qwen35.json").read_text())
    for source in config["sources"]:
        manifest = Path(source["dataset"], "manifest.jsonl")
        if manifest.exists():
            known = {
                json.loads(line)["split_group"]
                for line in manifest.read_text().splitlines()
            }
            assert not known.intersection(parents)


def test_optimized_resume_preserves_effective_batch_and_global_step_target():
    config = json.loads(Path("recipes/visual-followup-500-qwen35.json").read_text())
    config.update(
        training_module="cad_env.learning.train_optimized",
        batch_size=2,
        train_minutes=85,
        liger_layers=True,
        liger_profile="swiglu",
        resume_from_checkpoint="artifacts/visual-followup-500-v1/sft/checkpoint-100",
    )
    command = training_command(config)
    assert command[2] == "cad_env.learning.train_optimized"
    assert (
        int(command[command.index("--batch-size") + 1])
        * int(command[command.index("--gradient-accumulation") + 1])
        == 8
    )
    assert command[command.index("--max-steps") + 1] == "500"
    assert command[command.index("--max-train-minutes") + 1] == "85"
    assert command[command.index("--liger-profile") + 1] == "swiglu"
    assert "--resume-from-checkpoint" in command
