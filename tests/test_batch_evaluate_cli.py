import json
from types import SimpleNamespace

import pytest

from cad_env.learning import batch_evaluate


@pytest.mark.parametrize("condition", ["full", "no_goal", "blank_goal", "wrong_goal"])
def test_image_condition_reaches_native_protocol(monkeypatch, tmp_path, condition):
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_evaluate",
            "--model",
            "model",
            "--dataset",
            "data",
            "--output",
            str(tmp_path),
            "--workers",
            "8",
            "--image-condition",
            condition,
        ],
    )
    monkeypatch.setattr(
        batch_evaluate,
        "BatchedHFPolicy",
        lambda *a, **k: SimpleNamespace(identity={"model": "test"}),
    )

    class Batcher:
        def __init__(self, *a, **k):
            self.stats = {}

        def close(self):
            pass

    monkeypatch.setattr(batch_evaluate, "PolicyBatcher", Batcher)
    received = {}

    def evaluate(*a, **kwargs):
        received.update(kwargs)
        return {"metrics": {"success_at_1": 1}}

    monkeypatch.setattr(batch_evaluate, "evaluate_policy", evaluate)
    batch_evaluate.main()
    assert received["image_condition"] == condition and received["workers"] == 8
    assert received["policy_protocol"]["temperature"] == 0
    assert (
        json.loads((tmp_path / "batch-execution.json").read_text())["episode_workers"]
        == 8
    )


def test_invalid_worker_count_is_rejected_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_evaluate",
            "--model",
            "model",
            "--dataset",
            "data",
            "--output",
            str(tmp_path),
            "--workers",
            "0",
        ],
    )

    def load(*a, **k):
        raise AssertionError("model must not load")

    monkeypatch.setattr(batch_evaluate, "BatchedHFPolicy", load)
    with pytest.raises(SystemExit) as exc:
        batch_evaluate.main()
    assert exc.value.code == 2
