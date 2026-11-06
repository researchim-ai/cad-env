import json

import pytest
import torch

from cad_env.learning.kernel_results import summarize


@pytest.mark.parametrize("candidate_gradient,passed", [(2.0, True), (4.0, False)])
def test_kernel_gate_compares_identical_microbatch_profiles(
    tmp_path, candidate_gradient, passed
):
    profiles = {
        "baseline": (1.0, 10.0),
        "baseline_batch2": (2.0, 12.0),
        "fused": (1.0, 7.0),
        "fused_batch2": (candidate_gradient, 6.0),
    }
    status = {"status": "complete", "results": {}}
    for name, (gradient, seconds) in profiles.items():
        root = tmp_path / name
        (root / "checkpoint-8").mkdir(parents=True)
        torch.save(
            {
                "visual.lora_B": torch.tensor([gradient]),
                "language.lora_B": torch.tensor([gradient]),
            },
            root / "first-gradients.pt",
        )
        (root / "checkpoint-8/trainer_state.json").write_text(
            json.dumps({"log_history": [{"loss": 0.2}]})
        )
        status["results"][name] = {
            "status": "complete",
            "median_warm_step_seconds": seconds,
            "peak_allocated_gib": 12.0,
        }
    (tmp_path / "status.json").write_text(json.dumps(status))
    result = summarize(tmp_path)
    assert result["results"]["fused_batch2"]["gradient_reference"] == "baseline_batch2"
    assert result["results"]["fused_batch2"]["correctness_gate"] is passed
    assert result["recommended"] == ("fused_batch2" if passed else "fused")
