"""Optional SFT loss weighting for numeric CAD arguments.

JSON scaffolding can dominate token accuracy while the one circle-index token
is wrong. This loss weights ASCII integer tokens in assistant completions.
It does not expose target labels in prompts or modify native evaluation.
"""

import math
import re


class NumericCompletionLoss:
    def __init__(self, tokenizer, weight=8.0):
        if not math.isfinite(weight) or weight < 1:
            raise ValueError("Numeric token weight must be finite and at least one")
        self.numeric_ids = sorted(
            {i for t, i in tokenizer.get_vocab().items() if re.fullmatch(r"[0-9]+", t)}
        )
        if not self.numeric_ids:
            raise ValueError("No numeric tokens found")
        self.weight = weight
        self.numeric_tokens = 0
        self.numeric_correct = 0
        self.supervised_tokens = 0

    def __call__(self, outputs, labels, num_items_in_batch=None):
        import torch
        import torch.nn.functional as F

        targets = labels[..., 1:]
        valid = targets != -100
        if not valid.any():
            raise ValueError("No supervised next-token targets")
        # Gather completion positions before promoting logits to float32. Images
        # and public prompt tokens contribute neither loss nor statistics.
        logits = outputs.logits[..., :-1, :][valid].float()
        targets = targets[valid]
        ids = torch.tensor(self.numeric_ids, device=targets.device)
        numeric = torch.isin(targets, ids)
        weights = 1.0 + numeric.float() * (self.weight - 1.0)
        token_loss = F.cross_entropy(logits, targets, reduction="none")
        denominator = (
            num_items_in_batch if num_items_in_batch is not None else targets.numel()
        )
        with torch.no_grad():
            self.numeric_tokens += int(numeric.sum())
            self.numeric_correct += int(
                ((logits.argmax(-1) == targets) & numeric).sum()
            )
            self.supervised_tokens += targets.numel()
        return (token_loss * weights).sum() / denominator

    def summary(self):
        return {
            "numeric_token_weight": self.weight,
            "numeric_vocabulary_size": len(self.numeric_ids),
            "supervised_tokens": self.supervised_tokens,
            "numeric_tokens": self.numeric_tokens,
            "teacher_forced_numeric_accuracy": self.numeric_correct
            / self.numeric_tokens
            if self.numeric_tokens
            else None,
            "note": "Training token diagnostic; not CAD or visual success",
        }
