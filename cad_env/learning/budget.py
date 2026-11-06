"""Soft wall-time ceiling checked at optimizer-step boundaries, with durable saves."""

import json
import math
import time
from pathlib import Path


def make_walltime_guard(seconds, *, progress_path=None, clock=time.monotonic):
    from transformers import TrainerCallback

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive finite optimizer wall-time budget required")

    class WalltimeGuard(TrainerCallback):
        def __init__(self):
            self.started = None
            self.stopped = False
            self.steps = 0

        def summary(self):
            elapsed = (
                max(0.0, clock() - self.started) if self.started is not None else 0.0
            )
            return {
                "limit_seconds": seconds,
                "elapsed_seconds": elapsed,
                "budget_stopped": self.stopped,
                "optimizer_steps": self.steps,
                "scope": "Optimizer loop; checked after each step, excludes model loading and data preflight",
            }

        def persist(self):
            if progress_path is None:
                return
            path = Path(progress_path)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.summary(), indent=2))
            tmp.replace(path)

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = clock()
            self.steps = state.global_step
            self.persist()

        def on_step_end(self, args, state, control, **kwargs):
            self.steps = state.global_step
            if self.summary()["elapsed_seconds"] >= seconds:
                self.stopped = True
                control.should_save = True
                control.should_training_stop = True
            self.persist()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.steps = state.global_step
            self.persist()

    return WalltimeGuard()
