import pytest
import torch
from moshi.models.lm import LMOutput
from torch.nn import functional as F

from finetune.distillation.config import LossConfig
from finetune.distillation.losses import (
    HiddenMatcher,
    masked_ce,
    masked_kl,
    prediction_losses,
)


def test_invalid_nan_logits_are_removed_before_loss_and_backward():
    student = torch.tensor(
        [[[1.0, 2.0], [float("nan"), float("nan")]]], requires_grad=True
    )
    teacher = torch.tensor(
        [[[2.0, 1.0], [float("nan"), float("nan")]]], requires_grad=True
    )
    mask = torch.tensor([[True, False]])
    loss = masked_kl(student, teacher, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.equal(student.grad[0, 1], torch.zeros(2))
    assert teacher.grad is None


def test_identical_distributions_have_zero_kl():
    logits = torch.tensor([[[2.0, -1.0, 0.5]]], requires_grad=True)
    loss = masked_kl(logits, logits.detach(), torch.tensor([[True]]), temperature=2.0)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_temperature_scaled_forward_kl_and_weighted_normalization():
    student = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
    weights = torch.tensor([3.0, 1.0])
    temperature = 2.0
    p = F.softmax(teacher / temperature, dim=-1)
    expected = (
        (
            (p * (p.log() - F.log_softmax(student / temperature, dim=-1))).sum(-1)
            * weights
        ).sum()
        / weights.sum()
        * temperature**2
    )
    actual = masked_kl(
        student,
        teacher,
        torch.ones(2, dtype=torch.bool),
        temperature=temperature,
        weights=weights,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["ce", "kl"])
def test_empty_valid_stream_is_differentiable_zero(kind):
    logits = torch.full((2, 3), float("nan"), requires_grad=True)
    mask = torch.zeros(2, dtype=torch.bool)
    if kind == "ce":
        loss = masked_ce(logits, torch.tensor([-999, -999]), mask)
    else:
        loss = masked_kl(logits, logits.detach(), mask)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_zero_weight_nan_position_and_invalid_target_are_ignored():
    logits = torch.tensor(
        [[2.0, 0.0], [float("nan"), float("nan")]], requires_grad=True
    )
    target = torch.tensor([0, -999])
    mask = torch.ones(2, dtype=torch.bool)
    loss = masked_ce(logits, target, mask, weights=torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(loss, F.cross_entropy(logits[:1], target[:1]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "weights", [torch.tensor([-1.0]), torch.tensor([float("nan")])]
)
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match="Weights"):
        masked_ce(
            torch.zeros(1, 2),
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            weights=weights,
        )


def test_nonfinite_valid_logits_raise():
    with pytest.raises(ValueError, match="Valid logits"):
        masked_kl(
            torch.full((1, 2), float("nan")), torch.zeros(1, 2), torch.tensor([True])
        )


def test_hidden_matching_identity_masks_nans_and_freezes_teacher():
    matcher = HiddenMatcher(2, 2, 1)
    with torch.no_grad():
        matcher.projections[0].weight.copy_(torch.eye(2))
    student = torch.tensor(
        [[[1.0, 2.0], [float("nan"), float("nan")]]], requires_grad=True
    )
    teacher = torch.tensor(
        [[[2.0, 1.0], [float("nan"), float("nan")]]], requires_grad=True
    )
    loss = matcher([student], [teacher], torch.tensor([[True, False]]))
    loss.backward()
    assert loss.item() > 0
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert matcher.projections[0].weight.grad.abs().sum() > 0


def test_hidden_identity_is_zero_after_rms_normalization():
    matcher = HiddenMatcher(2, 2, 1)
    with torch.no_grad():
        matcher.projections[0].weight.copy_(torch.eye(2))
    state = torch.tensor([[[1.0, 2.0]]])
    loss = matcher([state], [state * 3], torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_audio_codebook_weights_are_normalized_and_zero_weights_ignore_nans():
    text = torch.zeros(1, 1, 1, 4, requires_grad=True)
    audio = torch.zeros(1, 8, 1, 2)
    audio[:, 0, :, 0] = 2
    audio[:, 1:] = float("nan")
    audio.requires_grad_()
    student = LMOutput(
        audio,
        torch.ones(1, 8, 1, dtype=torch.bool),
        text,
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    teacher = LMOutput(audio.detach(), student.mask, text.detach(), student.text_mask)
    config = LossConfig(
        text_kl_weight=0,
        audio_kl_weight=0,
        hidden_weight=0,
        audio_codebook_weights=[10.0] + [0.0] * 7,
    )
    total, parts = prediction_losses(
        student,
        teacher,
        torch.zeros(1, 17, 1, dtype=torch.long),
        config,
        padding_ids=set(),
    )
    expected = F.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0]))
    torch.testing.assert_close(parts["audio_ce"], expected)
    total.backward()
    assert torch.isfinite(audio.grad).all()


def test_kl_uses_intersection_of_teacher_and_student_masks():
    text = torch.tensor(
        [[[[1.0, 0.0], [float("nan"), float("nan")]]]], requires_grad=True
    )
    audio = torch.zeros(1, 8, 2, 2, requires_grad=True)
    student = LMOutput(
        audio,
        torch.ones(1, 8, 2, dtype=torch.bool),
        text,
        torch.ones(1, 1, 2, dtype=torch.bool),
    )
    teacher_text = torch.tensor([[[[0.0, 1.0], [float("nan"), float("nan")]]]])
    teacher = LMOutput(
        audio.detach(), student.mask, teacher_text, torch.tensor([[[True, False]]])
    )
    config = LossConfig(data_weight=0, hidden_weight=0)
    total, parts = prediction_losses(
        student,
        teacher,
        torch.zeros(1, 17, 2, dtype=torch.long),
        config,
        padding_ids=set(),
    )
    expected = masked_kl(
        text[..., :1, :],
        teacher_text[..., :1, :],
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(parts["text_kl"], expected)
    total.backward()
    assert torch.isfinite(text.grad).all()
