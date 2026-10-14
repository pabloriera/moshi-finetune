"""Mask before softmax: Moshi fills invalid delayed positions with NaNs."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import LossConfig, finite_number


def _valid(logits, mask, weights=None):
    if mask.dtype != torch.bool or logits.shape[:-1] != mask.shape:
        raise ValueError("Boolean mask must match all non-vocabulary logits dimensions")
    if weights is None:
        weights = torch.ones_like(mask, dtype=torch.float32)
    if (
        weights.shape != mask.shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("Weights must be finite, nonnegative, and match the mask")
    valid = mask & (weights > 0)
    return valid, weights[valid].float()


def masked_kl(student, teacher, mask, *, temperature=1.0, weights=None):
    finite_number(temperature, "temperature", positive=True)
    if student.shape != teacher.shape:
        raise ValueError("Teacher and student logits must have identical shapes")
    valid, selected_weights = _valid(student, mask, weights)
    selected = student[valid].float()
    if selected.numel() == 0:
        return selected.sum()  # Differentiable zero; never multiply NaN by zero.
    target = teacher.detach()[valid].float()
    if not torch.isfinite(selected).all() or not torch.isfinite(target).all():
        raise ValueError("Valid logits must be finite")
    per_token = F.kl_div(
        F.log_softmax(selected / temperature, dim=-1),
        F.log_softmax(target / temperature, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(-1)
    return (
        (per_token * selected_weights).sum() / selected_weights.sum() * temperature**2
    )


def masked_ce(logits, target, mask, *, weights=None):
    if target.shape != mask.shape or target.dtype != torch.long:
        raise ValueError("Integer targets must match mask shape")
    valid, selected_weights = _valid(logits, mask, weights)
    selected = logits[valid].float()
    if selected.numel() == 0:
        return selected.sum()
    if not torch.isfinite(selected).all():
        raise ValueError("Valid logits must be finite")
    per_token = F.cross_entropy(selected, target[valid], reduction="none")
    return (per_token * selected_weights).sum() / selected_weights.sum()


class HiddenMatcher(nn.Module):
    """Learn student -> fixed teacher projections; not PCA weight initialization."""

    def __init__(self, teacher_dim: int, student_dim: int, num_pairs: int):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(student_dim, teacher_dim, bias=False) for _ in range(num_pairs)
        )

    def forward(self, student_states, teacher_states, mask):
        if len(student_states) != len(self.projections) or len(teacher_states) != len(
            self.projections
        ):
            raise ValueError(
                "Hidden-state lists must match the configured layer mapping"
            )
        losses = []
        for student, teacher, projection in zip(
            student_states, teacher_states, self.projections
        ):
            if (
                student.shape[:-1] != mask.shape
                or teacher.shape[:-1] != mask.shape
                or mask.dtype != torch.bool
            ):
                raise ValueError("Hidden masks must match batch/time dimensions")
            selected_student = student[mask]
            if selected_student.numel() == 0:
                losses.append(selected_student.sum())
                continue
            selected = projection(selected_student.to(projection.weight.dtype)).float()
            target = teacher.detach()[mask].float()
            if not torch.isfinite(selected).all() or not torch.isfinite(target).all():
                raise ValueError("Valid hidden states must be finite")
            selected = (
                selected
                / selected.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
            )
            target = (
                target / target.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
            )
            losses.append(F.mse_loss(selected, target))
        return torch.stack(losses).mean()


def prediction_losses(student, teacher, codes, config: LossConfig, *, padding_ids):
    """Normalize each text/audio objective by its own weighted valid-token count."""
    text_mask = student.text_mask & teacher.text_mask
    audio_mask = student.mask & teacher.mask
    text_weights = torch.ones_like(text_mask, dtype=torch.float32)
    for token in padding_ids:
        text_weights[codes[:, :1] == token] = config.text_padding_weight
    audio_weights = torch.tensor(
        config.audio_codebook_weights, device=codes.device
    ).view(1, 8, 1)
    audio_weights = audio_weights.expand_as(audio_mask)
    # This zero is safe even when the first logits contain invalid NaNs.
    zero = student.text_logits.reshape(-1)[:0].sum()
    text_ce = audio_ce = text_kl = audio_kl = zero
    if config.data_weight:
        text_ce = masked_ce(
            student.text_logits, codes[:, :1], text_mask, weights=text_weights
        )
        audio_ce = masked_ce(
            student.logits, codes[:, 1:9], audio_mask, weights=audio_weights
        )
    if config.text_kl_weight:
        text_kl = masked_kl(
            student.text_logits,
            teacher.text_logits,
            text_mask,
            temperature=config.text_temperature,
            weights=text_weights,
        )
    if config.audio_kl_weight:
        audio_kl = masked_kl(
            student.logits,
            teacher.logits,
            audio_mask,
            temperature=config.audio_temperature,
            weights=audio_weights,
        )
    total = (
        config.data_weight * (text_ce + audio_ce)
        + config.text_kl_weight * text_kl
        + config.audio_kl_weight * audio_kl
    )
    return total, {
        "text_ce": text_ce,
        "audio_ce": audio_ce,
        "text_kl": text_kl,
        "audio_kl": audio_kl,
    }
