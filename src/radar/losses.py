"""Training losses for RADAR."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import (
    MONTH_END_DAYS,
    MONTH_START_DAYS,
    MONTH_WIDTH_DAYS,
    positive_hazards,
)


def competing_risk_nll(
    raw_hazards: torch.Tensor,
    followup_days: torch.Tensor,
    event_types: torch.Tensor,
    risk_masks: torch.Tensor,
) -> torch.Tensor:
    """Exact-exposure piecewise-exponential competing-risk NLL.

    Event types are 0 for censored/event-free, 1 for AF/flutter, and 2 for
    death. Partially observed follow-up contributes its observed exposure.
    """
    hazards = positive_hazards(raw_hazards)
    followup = followup_days.float().clamp(0.0, float(MONTH_END_DAYS[-1]))
    starts = followup.new_tensor(MONTH_START_DAYS)[None, :]
    widths = followup.new_tensor(MONTH_WIDTH_DAYS)[None, :]
    exposure = (followup[:, None] - starts).clamp_min(0.0).minimum(widths)
    integrated = (hazards.sum(dim=-1) * exposure).sum(dim=1)

    event = event_types.long()
    if torch.any((event < 0) | (event > 2)):
        raise ValueError("event_types must contain only 0, 1, or 2")
    event_mask = event.gt(0) & risk_masks.bool()
    interval = torch.bucketize(
        followup, followup.new_tensor(MONTH_END_DAYS), right=False
    ).clamp_max(len(MONTH_END_DAYS) - 1)
    cause = (event - 1).clamp_min(0)
    selected = hazards[
        torch.arange(len(hazards), device=hazards.device), interval, cause
    ]
    losses = integrated - event_mask.to(integrated.dtype) * torch.log(selected)
    weights = risk_masks.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def domain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ICD coding-family classifier loss; labels below zero are ignored."""
    valid = labels.ge(0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid].to(dtype=logits.dtype)
    )


def domain_schedule(
    step: int,
    total_steps: int,
    maximum: float = 0.03,
    start_fraction: float = 0.05,
) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    if progress <= start_fraction:
        return 0.0
    scaled = (progress - start_fraction) / max(1.0 - start_fraction, 1e-8)
    return float(maximum * (2.0 / (1.0 + math.exp(-10.0 * scaled)) - 1.0))
