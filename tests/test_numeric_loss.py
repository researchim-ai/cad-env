"""Check that the numeric loss preserves causal masking and accumulation scale."""

from types import SimpleNamespace

import pytest


def test_numeric_loss_weights_critical_tokens_and_never_prompt_tokens():
    torch = pytest.importorskip("torch")
    from cad_env.learning.numeric_loss import NumericCompletionLoss

    class Tokenizer:
        def get_vocab(self):
            return {"scaffold": 0, "0": 1, "1": 2, "image": 3}

    loss = NumericCompletionLoss(Tokenizer(), weight=8)
    logits = torch.zeros((1, 4, 4), requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 0]])
    result = loss(SimpleNamespace(logits=logits), labels)
    assert result.item() == pytest.approx(4.5 * torch.log(torch.tensor(4.0)).item())
    result.backward()
    assert logits.grad[0, 0].abs().sum() == 0  # prompt/image position
    assert logits.grad[0, 1].abs().sum() == pytest.approx(
        8 * logits.grad[0, 2].abs().sum()
    )
    assert logits.grad[0, 3].abs().sum() == 0  # no future target
    assert loss.summary()["numeric_tokens"] == 1
    assert loss.summary()["supervised_tokens"] == 2
    accumulated = loss(
        SimpleNamespace(logits=logits.detach()), labels, num_items_in_batch=4
    )
    assert accumulated.item() == pytest.approx(result.item() / 2)
