"""Bounded, deferred same-data 4B throughput probes on physical GPU 1 only."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(mode, output, steps=4):
    import torch
    import trl
    from transformers import TrainerCallback

    torch.manual_seed(42)

    class Timing(TrainerCallback):
        def __init__(self):
            self.times = []

        def on_step_begin(self, *args, **kwargs):
            torch.cuda.synchronize()
            self.started = time.monotonic()

        def on_step_end(self, *args, **kwargs):
            torch.cuda.synchronize()
            self.times.append(time.monotonic() - self.started)
            Path(output, "step-times.json").write_text(json.dumps(self.times))

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if state.global_step == 0:
                torch.save(
                    {
                        n: p.grad.detach().float().cpu()
                        for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None
                    },
                    Path(output, "first-gradients.pt"),
                )

    original = trl.SFTTrainer

    class TimedTrainer(original):
        def __init__(self, *args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), Timing()]
            super().__init__(*args, **kwargs)

    trl.SFTTrainer = TimedTrainer
    from .visual_followup import training_command

    config = json.loads(Path("recipes/visual-followup-500-qwen35.json").read_text())
    batch = 4 if mode.endswith("batch4") else 2 if mode.endswith("batch2") else 1
    argv = training_command(config)[3:]
    for flag, value in {
        "--output": output,
        "--max-steps": str(steps),
        "--warmup-steps": "0",
        "--save-steps": str(steps),
        "--save-total-limit": "1",
        "--max-train-minutes": "4",
        "--batch-size": str(batch),
        "--gradient-accumulation": str(8 // batch),
    }.items():
        argv[argv.index(flag) + 1] = value
    argv += ["--max-train-rows", str(steps * 8)]
    sys.argv = [sys.argv[0], *argv]
    if mode.startswith("baseline"):
        from .train_modern import main
    else:
        from .train_optimized import main

        if "liger" in mode:
            sys.argv.append("--liger-layers")
        for profile in ["rms", "swiglu"]:
            if profile in mode:
                sys.argv += ["--liger-profile", profile]
        if "nogc" in mode:
            sys.argv += ["--no-gradient-checkpointing"]
        if "hybrid" in mode:
            sys.argv += ["--checkpoint-policy", "linear", "--memory-preflight"]
    main()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--worker",
        choices=[
            "baseline",
            "fused",
            "fused_liger",
            "fused_liger_batch2",
            "fused_liger_rms",
            "fused_liger_swiglu",
            "baseline_batch2",
            "baseline_batch4",
            "fused_batch2",
            "fused_batch4",
            "fused_liger_swiglu_batch2",
            "fused_nogc",
            "fused_hybrid",
        ],
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "fused", "fused_liger", "fused_liger_batch2"],
    )
    p.add_argument("--probe-steps", type=int, default=4)
    p.add_argument(
        "--wait-for",
        help="Wait for terminal experiment status and idle GPU, at most four hours",
    )
    args = p.parse_args()
    valid_modes = {
        "baseline",
        "baseline_batch2",
        "baseline_batch4",
        "fused",
        "fused_batch2",
        "fused_batch4",
        "fused_liger",
        "fused_liger_batch2",
        "fused_liger_rms",
        "fused_liger_swiglu",
        "fused_liger_swiglu_batch2",
        "fused_nogc",
        "fused_hybrid",
    }
    if not 4 <= args.probe_steps <= 8 or not set(args.modes) <= valid_modes:
        p.error("Use known profiles and four to eight probe steps")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU 1 only")
    if args.worker:
        worker(args.worker, args.output, args.probe_steps)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "waiting",
        "results": {},
        "scope": f"{args.probe_steps}-step speed probes, not quality training; first step excluded from median",
    }

    def persist():
        (out / "status.json").write_text(json.dumps(state, indent=2))

    persist()
    waiting_deadline = time.monotonic() + 4 * 3600
    try:
        while True:
            status = (
                json.loads(Path(args.wait_for).read_text())
                if args.wait_for
                else {"status": "complete"}
            )
            used = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        "1",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            if status["status"] in {"complete", "failed"} and used < 250:
                break
            if time.monotonic() >= waiting_deadline:
                raise TimeoutError("GPU did not become idle; no probes launched")
            time.sleep(30)
        deadline = time.monotonic() + 30 * 60
        for mode in args.modes:
            state["status"] = mode
            persist()
            with (out / (mode + ".log")).open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "cad_env.learning.kernel_benchmark",
                        "--worker",
                        mode,
                        "--output",
                        str(out / mode),
                        "--probe-steps",
                        str(args.probe_steps),
                    ],
                    stdout=log,
                    check=False,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, deadline - time.monotonic()),
                )
            if result.returncode:
                state["results"][mode] = {
                    "status": "failed",
                    "returncode": result.returncode,
                }
                persist()
                continue
            manifest = json.loads((out / mode / "cad-training-run.json").read_text())
            times = json.loads((out / mode / "step-times.json").read_text())
            if (
                len(times) != args.probe_steps
                or not manifest["completed_planned_steps"]
            ):
                raise RuntimeError("Incomplete speed probe")
            state["results"][mode] = {
                "status": "complete",
                "step_seconds": times,
                "median_warm_step_seconds": statistics.median(times[1:]),
                "peak_allocated_gib": manifest["peak_allocated_gib"],
                "gradient_signal": manifest["gradient_signal"],
                "visual_gradient_signal": manifest["visual_gradient_signal"],
            }
            persist()
        state["status"] = "complete"
        persist()
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        persist()
        raise


if __name__ == "__main__":
    main()
