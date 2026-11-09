import json
import os
import time
from pathlib import Path

import pytest

from cad_env.learning import release_trial as trial


def interrupted(tmp_path):
    root = tmp_path / "trial"
    checkpoint = root / "candidate/sft/checkpoint-50"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":50}')
    (checkpoint.parent / "budget-progress.json").write_text('{"optimizer_steps":67}')
    state = {
        "status": "candidate_train",
        "limits": {"total_minutes": 165},
        "actual_steps": {"control": 200},
        "stages": {
            "candidate_train": {
                "command": [
                    "python",
                    "-m",
                    "cad_env.learning.train_optimized",
                    "--output",
                    str(checkpoint.parent),
                    "--max-steps",
                    "200",
                ]
            }
        },
    }
    (root / "status.json").write_text(json.dumps(state))
    (root / "source-freeze.json").write_text("{}")
    now = time.time()
    os.utime(root / "source-freeze.json", (now - 3600, now - 3600))
    return root


def test_resume_preserves_completed_control_and_remaining_budget(tmp_path, monkeypatch):
    root = interrupted(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert 0 < kwargs["timeout"] <= 105 * 60
        if command[2] == "cad_env.learning.train_optimized":
            target = Path(command[command.index("--output") + 1])
            target.mkdir()
            (target / "cad-training-run.json").write_text(
                '{"status":"complete","optimizer_steps":200}'
            )

    monkeypatch.setattr(trial.subprocess, "run", run)
    monkeypatch.setattr(
        trial, "finish_results", lambda out, state: state.update(status="complete")
    )
    trial.resume_trial(root)
    report = json.loads((root / "status.json").read_text())
    assert report["status"] == "complete"
    assert report["actual_steps"] == {"control": 200, "candidate": 200}
    assert commands[0][-2:] == [
        "--resume-from-checkpoint",
        str(root / "candidate/sft/checkpoint-50"),
    ]
    assert len(commands) == 12  # candidate train/merge + ten evaluations
    assert (root / "interruption-evidence/status.json").exists()
    assert (
        json.loads((root / "interruption-evidence/status.json").read_text())["status"]
        == "candidate_train"
    )


def test_resume_rejects_changed_trainer(tmp_path):
    root = interrupted(tmp_path)
    source = tmp_path / "trainer.py"
    source.write_text("changed")
    (root / "source-freeze.json").write_text(json.dumps({str(source): "wronghash"}))
    with pytest.raises(RuntimeError, match="Source drift"):
        trial.resume_trial(root)


def test_resume_rejects_completed_trial(tmp_path):
    root = interrupted(tmp_path)
    (root / "status.json").write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match="interrupted"):
        trial.resume_trial(root)
