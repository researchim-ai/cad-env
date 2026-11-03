"""Fail a training run when logs reveal numerical failure or absent learning signal."""

import math


def make_numerical_guard(require_signal=False, max_zero_steps=20):
    from transformers import TrainerCallback

    class NumericalGuard(TrainerCallback):
        def __init__(self):
            self.norms = []
            self.consecutive_zero = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            for name in ("loss", "grad_norm"):
                if name in logs and not math.isfinite(float(logs[name])):
                    raise FloatingPointError("Non-finite training " + name)
            if "grad_norm" in logs:
                norm = float(logs["grad_norm"])
                self.norms.append(norm)
                self.consecutive_zero = self.consecutive_zero + 1 if norm == 0 else 0
                if require_signal and self.consecutive_zero >= max_zero_steps:
                    raise RuntimeError(
                        "No gradient signal; stop and inspect loss masks or GRPO group variance"
                    )

        def on_train_end(self, args, state, control, **kwargs):
            if require_signal and not any(value > 0 for value in self.norms):
                raise RuntimeError("Training ended without a nonzero logged gradient")

        def summary(self):
            return {
                "logged_steps": len(self.norms),
                "nonzero_steps": sum(v > 0 for v in self.norms),
                "max_grad_norm": max(self.norms, default=None),
            }

    return NumericalGuard()
