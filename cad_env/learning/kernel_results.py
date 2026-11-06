"""Compare paired first-update gradients before choosing a throughput profile."""

import argparse
import json
from pathlib import Path


def summarize(root):
    import torch

    root = Path(root)
    status = json.loads((root / "status.json").read_text())
    if (
        status["status"] != "complete"
        or status["results"].get("baseline", {}).get("status") != "complete"
    ):
        raise ValueError("Completed baseline benchmark required")

    def first_loss(mode):
        checkpoint = max(
            (root / mode).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        return next(row["loss"] for row in state["log_history"] if "loss" in row)

    baseline_time = status["results"]["baseline"]["median_warm_step_seconds"]
    baseline_loss = first_loss("baseline")
    results = {}
    for mode, timing in status["results"].items():
        if mode.startswith("baseline") or timing["status"] != "complete":
            continue
        suffix = (
            "_batch4"
            if mode.endswith("batch4")
            else "_batch2"
            if mode.endswith("batch2")
            else ""
        )
        reference_mode = "baseline" + suffix
        if status["results"].get(reference_mode, {}).get("status") != "complete":
            results[mode] = {
                **timing,
                "correctness_gate": False,
                "eligible": False,
                "reason": "No complete same-microbatch reference: " + reference_mode,
            }
            continue
        reference = torch.load(
            root / reference_mode / "first-gradients.pt",
            map_location="cpu",
            weights_only=True,
        )
        candidate = torch.load(
            root / mode / "first-gradients.pt", map_location="cpu", weights_only=True
        )
        if candidate.keys() != reference.keys():
            raise ValueError("Different trainable gradient keys")
        errors = {}
        for group in ["all", "visual"]:
            names = [n for n in reference if group == "all" or "visual" in n]
            squared_delta = sum(
                float((candidate[n] - reference[n]).square().sum()) for n in names
            )
            squared_reference = sum(float(reference[n].square().sum()) for n in names)
            if squared_reference <= 0:
                raise ValueError("Reference gradient signal missing: " + group)
            errors[group] = (squared_delta / squared_reference) ** 0.5
        loss = first_loss(mode)
        matched_loss = first_loss(reference_mode)
        loss_error = abs(loss - matched_loss) / max(abs(matched_loss), 1e-8)
        speedup = baseline_time / timing["median_warm_step_seconds"]
        correct = errors["all"] < 0.03 and errors["visual"] < 0.05 and loss_error < 0.01
        memory_headroom = timing["peak_allocated_gib"] <= 20.5
        results[mode] = {
            **timing,
            "first_loss": loss,
            "gradient_reference": reference_mode,
            "relative_loss_error": loss_error,
            "relative_gradient_l2_error": errors,
            "speedup": speedup,
            "correctness_gate": correct,
            "memory_headroom_gate": memory_headroom,
            "eligible": correct and memory_headroom and speedup > 1.10,
        }
    eligible = [name for name, result in results.items() if result["eligible"]]
    recommended = (
        max(eligible, key=lambda name: results[name]["speedup"]) if eligible else None
    )
    if (
        "fused" in eligible
        and recommended != "fused"
        and results[recommended]["speedup"] / results["fused"]["speedup"] < 1.05
    ):
        recommended = "fused"
    return {
        "baseline_first_loss": baseline_loss,
        "baseline_step_seconds": baseline_time,
        "results": results,
        "recommended": recommended,
        "selection_rule": "Prefer the simpler fused-only path unless optional kernels add at least 5% speed; this practical threshold is not a statistical significance test",
        "scope": "Short same-data, same-seed pilot; first clipped gradients compared only at the same microbatch size to isolate kernel effects from BF16 batching effects. Does not establish full-corpus batch memory safety or better CAD accuracy.",
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--benchmark", required=True)
    args = p.parse_args()
    result = summarize(args.benchmark)
    Path(args.benchmark, "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
