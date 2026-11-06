from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("denominator", [None, 19])
def test_fused_weighting_matches_loss_gradients_and_masks(denominator):
    import torch

    from cad_env.learning.fused_numeric import fused_numeric_loss
    from cad_env.learning.numeric_loss import NumericCompletionLoss

    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"text": 0, "1": 1, "2": 2, "image": 3}
    )
    torch.manual_seed(41)
    a = torch.randn(2, 7, 4, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    labels = torch.tensor(
        [[-100, -100, 1, 0, 2, 0, -100], [-100, 2, 0, 1, -100, -100, -100]]
    )
    original = NumericCompletionLoss(tokenizer)
    fused = NumericCompletionLoss(tokenizer)
    expected = original(SimpleNamespace(logits=a), labels, denominator)
    targets = labels[:, 1:]
    mask = targets != -100
    lp = (
        b[:, :-1]
        .log_softmax(-1)
        .gather(-1, targets.clamp_min(0).unsqueeze(-1))
        .squeeze(-1)
    )
    outputs = SimpleNamespace(
        log_probs=lp.masked_fill(~mask, 0),
        label_mask=mask,
        is_top1=(b[:, :-1].argmax(-1) == targets) & mask,
    )
    actual = fused_numeric_loss(outputs, labels, fused, denominator)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(a.grad, b.grad)
    assert original.summary() == fused.summary()


def test_fused_rejects_misaligned_mask():
    import torch

    from cad_env.learning.fused_numeric import fused_numeric_loss

    outputs = SimpleNamespace(
        log_probs=torch.zeros(1, 2), label_mask=torch.ones(1, 2, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="alignment"):
        fused_numeric_loss(outputs, torch.tensor([[-100, -100, 1]]), None)


def test_trainer_uses_one_forward_and_window_denominator():
    from collections import defaultdict

    import torch

    from cad_env.learning.fused_numeric import trainer_class
    from cad_env.learning.numeric_loss import NumericCompletionLoss

    class Base:
        def __init__(self, compute_loss_func=None):
            assert compute_loss_func is None
            self.accelerator = SimpleNamespace(num_processes=1)
            self.aux_loss_enabled = False
            self._metrics = defaultdict(lambda: defaultdict(list))
            self._total_train_tokens = 0

    calls = []
    labels = torch.tensor([[-100, 1, 0]])
    log_probs = torch.tensor([[-2.0, -3.0]], requires_grad=True)

    class Model:
        training = True

        def __call__(self, **inputs):
            calls.append(inputs)
            return SimpleNamespace(
                log_probs=log_probs,
                label_mask=torch.ones(1, 2, dtype=torch.bool),
                is_top1=torch.zeros(1, 2, dtype=torch.bool),
                entropy=torch.ones(1, 2),
            )

    numeric = NumericCompletionLoss(SimpleNamespace(get_vocab=lambda: {"1": 1}))
    trainer = trainer_class(Base)(compute_loss_func=numeric)
    loss = trainer.compute_loss(
        Model(),
        {"labels": labels, "attention_mask": torch.ones(1, 3)},
        num_items_in_batch=8,
    )
    assert loss.item() == pytest.approx((2 * 8 + 3) / 8)
    loss.backward()
    assert len(calls) == 1 and calls[0]["fused_lm_head"]
    assert trainer.model_accepts_loss_kwargs
    torch.testing.assert_close(log_probs.grad, torch.tensor([[-1.0, -0.125]]))


def test_hybrid_checkpoint_policy_survives_repeated_trainer_enable():
    from cad_env.learning.fused_numeric import checkpoint_linear_layers_only

    layers = [
        SimpleNamespace(block_type=t, gradient_checkpointing=False)
        for t in ["linear_attention", "full_attention"]
    ]

    def enable(**kwargs):
        for layer in layers:
            layer.gradient_checkpointing = True

    base = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
        gradient_checkpointing_enable=enable,
    )
    checkpoint_linear_layers_only(SimpleNamespace(get_base_model=lambda: base))
    for _ in range(2):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        assert layers[0].gradient_checkpointing
        assert not layers[1].gradient_checkpointing
