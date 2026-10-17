"""The RADAR longitudinal diagnosis architecture.

RADAR represents every encounter as an unordered set of diagnosis codes,
updates a fixed bank of exponentially decayed memories, and uses monthly
queries to estimate cause-specific AF/flutter and death hazards.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


MONTH_END_DAYS = (
    30.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
    210.0,
    240.0,
    270.0,
    300.0,
    330.0,
    365.0,
)
MONTH_START_DAYS = (0.0, *MONTH_END_DAYS[:-1])
MONTH_WIDTH_DAYS = tuple(
    end - start for start, end in zip(MONTH_START_DAYS, MONTH_END_DAYS)
)
AF_CAUSE = 0
DEATH_CAUSE = 1


def _segment_softmax(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    """Compute a softmax over codes belonging to the same encounter."""
    if scores.numel() == 0:
        return scores
    expanded_ids = segment_ids[:, None].expand(-1, scores.shape[1])
    maxima = torch.full(
        (n_segments, scores.shape[1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima = maxima.scatter_reduce(
        0, expanded_ids, scores, reduce="amax", include_self=True
    )
    centered = scores - maxima.index_select(0, segment_ids)
    numerators = centered.exp()
    denominators = torch.zeros_like(maxima).scatter_add(0, expanded_ids, numerators)
    return numerators / denominators.index_select(0, segment_ids).clamp_min(1e-8)


class VisitSetEncoder(nn.Module):
    """Learned-query pooling over the unordered diagnosis set of each visit."""

    def __init__(self, config: "RadarConfig") -> None:
        super().__init__()
        d = config.embedding_dim
        self.embedding = nn.Embedding(config.vocab_size, d)
        self.value_norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.empty(config.visit_queries, d))
        self.output = nn.Sequential(
            nn.Linear(config.visit_queries * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        code_ids: torch.Tensor,
        code_offsets: torch.Tensor,
    ) -> torch.Tensor:
        n_encounters = int(code_offsets.numel() - 1)
        if n_encounters < 0:
            raise ValueError("code_offsets must contain at least one value")
        if n_encounters == 0:
            return self.embedding.weight.new_zeros((0, self.embedding.embedding_dim))
        lengths = code_offsets[1:] - code_offsets[:-1]
        if torch.any(lengths < 0):
            raise ValueError("code_offsets must be nondecreasing")
        if int(code_offsets[-1]) != int(code_ids.numel()):
            raise ValueError("final code offset does not match code_ids length")

        d = self.embedding.embedding_dim
        q = self.queries.shape[0]
        if code_ids.numel() == 0:
            pooled = self.embedding.weight.new_zeros((n_encounters, q, d))
        else:
            segment_ids = torch.repeat_interleave(
                torch.arange(n_encounters, device=code_ids.device), lengths
            )
            values = self.value_norm(self.embedding(code_ids))
            scores = values @ self.queries.transpose(0, 1) / math.sqrt(d)
            weights = _segment_softmax(scores, segment_ids, n_encounters)
            weighted = weights[:, :, None] * values[:, None, :]
            pooled_flat = values.new_zeros((n_encounters, q * d))
            pooled_flat = pooled_flat.index_add(
                0, segment_ids, weighted.reshape(-1, q * d)
            )
            pooled = pooled_flat.reshape(n_encounters, q, d)
        return self.output(pooled.reshape(n_encounters, q * d))


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(values: torch.Tensor, coefficient: float) -> torch.Tensor:
    return _GradientReversal.apply(values, coefficient)


@dataclass(frozen=True)
class RadarConfig:
    vocab_size: int
    embedding_dim: int = 96
    visit_queries: int = 2
    memory_channels: int = 8
    attention_heads: int = 4
    dropout: float = 0.1
    domain_hidden_dim: int = 64
    half_lives_days: tuple[float, ...] = (
        7.0,
        30.0,
        90.0,
        180.0,
        365.0,
        1095.0,
        3650.0,
        36500.0,
    )

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embedding_dim <= 0 or self.embedding_dim % self.attention_heads:
            raise ValueError(
                "embedding_dim must be positive and divisible by attention_heads"
            )
        if self.visit_queries <= 0 or self.memory_channels <= 0:
            raise ValueError("visit_queries and memory_channels must be positive")
        if len(self.half_lives_days) != self.memory_channels:
            raise ValueError("one initial half-life is required per memory channel")
        if any(value <= 0 for value in self.half_lives_days):
            raise ValueError("half-lives must be positive")
        if self.domain_hidden_dim <= 0:
            raise ValueError("domain_hidden_dim must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RadarOutput:
    raw_hazards: torch.Tensor
    domain_logits: torch.Tensor
    current_visit_representation: torch.Tensor | None = None
    attention: torch.Tensor | None = None


def positive_hazards(raw_hazards: torch.Tensor) -> torch.Tensor:
    if raw_hazards.ndim != 3 or raw_hazards.shape[-2:] != (12, 2):
        raise ValueError("raw_hazards must have shape [encounters, 12, 2]")
    return nn.functional.softplus(raw_hazards.float()) + 1e-8


def cumulative_incidence(
    raw_hazards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return AF CIF, death CIF, and event-free survival by month."""
    hazards = positive_hazards(raw_hazards)
    widths = hazards.new_tensor(MONTH_WIDTH_DAYS)[None, :]
    total = hazards.sum(dim=-1)
    conditional_event = -torch.expm1(-total * widths)
    survival_before = torch.exp(
        -torch.cumsum(
            torch.cat(
                (total[:, :1] * 0.0, total[:, :-1] * widths[:, :-1]),
                dim=1,
            ),
            dim=1,
        )
    )
    increments = (
        survival_before[:, :, None]
        * conditional_event[:, :, None]
        * hazards
        / total.clamp_min(1e-12)[:, :, None]
    )
    af_cif = torch.cumsum(increments[:, :, AF_CAUSE], dim=1)
    death_cif = torch.cumsum(increments[:, :, DEATH_CAUSE], dim=1)
    survival = torch.exp(-torch.cumsum(total * widths, dim=1))
    return af_cif, death_cif, survival


class RadarModel(nn.Module):
    """Parameter-efficient longitudinal competing-risk model."""

    def __init__(self, config: RadarConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.embedding_dim
        k = config.memory_channels

        self.visit_encoder = VisitSetEncoder(config)
        self.memory_value = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.memory_gate = nn.Linear(d, k)
        self.memory_channel_embedding = nn.Parameter(torch.empty(k, d))
        self.log_half_lives = nn.Parameter(
            torch.log(torch.tensor(config.half_lives_days, dtype=torch.float32))
        )
        self.demographic_encoder = nn.Sequential(
            nn.Linear(2, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.current_visit_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.interval_queries = nn.Parameter(torch.empty(12, d))
        self.readout_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(d)
        self.hazard_weight = nn.Parameter(torch.empty(12, 2, d))
        self.hazard_bias = nn.Parameter(torch.full((12, 2), -8.0))
        self.domain_classifier = nn.Sequential(
            nn.Linear(d + 3, config.domain_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.domain_hidden_dim, 1),
        )
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.memory_channel_embedding, std=0.02)
        nn.init.normal_(self.interval_queries, std=0.02)
        nn.init.normal_(self.hazard_weight, std=0.02)

    @property
    def half_lives_days(self) -> torch.Tensor:
        return self.log_half_lives.clamp(math.log(1.0), math.log(36500.0)).exp()

    def forward(
        self,
        *,
        code_ids: torch.Tensor,
        code_offsets: torch.Tensor,
        patient_ptr: torch.Tensor,
        event_days: torch.Tensor,
        ages: torch.Tensor,
        sexes: torch.Tensor,
        diagnosis_counts: torch.Tensor,
        domain_coefficient: float = 0.0,
        return_representation: bool = False,
        return_attention: bool = False,
    ) -> RadarOutput:
        visit_flat = self.visit_encoder(code_ids, code_offsets)
        n_encounters = visit_flat.shape[0]
        if any(
            tensor.numel() != n_encounters
            for tensor in (event_days, ages, sexes, diagnosis_counts)
        ):
            raise ValueError("encounter-level tensors must align")
        if code_offsets.numel() != n_encounters + 1:
            raise ValueError("code_offsets must delimit every encounter")
        if patient_ptr.numel() < 2 or int(patient_ptr[-1]) != n_encounters:
            raise ValueError("patient_ptr does not cover every encounter")
        lengths = patient_ptr[1:] - patient_ptr[:-1]
        if torch.any(lengths <= 0):
            raise ValueError("every patient must have at least one encounter")

        n_patients = int(lengths.numel())
        d = self.config.embedding_dim
        k = self.config.memory_channels
        state = visit_flat.new_zeros((n_patients, k, d))
        previous_days = event_days.new_zeros(n_patients)
        history_indices: list[torch.Tensor] = []
        history_values: list[torch.Tensor] = []
        half_lives = self.half_lives_days.to(device=visit_flat.device)

        for step in range(int(lengths.max())):
            active = torch.nonzero(lengths > step, as_tuple=False).flatten()
            encounter_indices = patient_ptr.index_select(0, active) + step
            current_days = event_days.index_select(0, encounter_indices)
            delta = (
                torch.zeros_like(current_days)
                if step == 0
                else (current_days - previous_days.index_select(0, active)).clamp_min(0)
            )
            decay = torch.exp(-math.log(2.0) * delta[:, None] / half_lives[None, :]).to(
                dtype=state.dtype
            )
            active_state = state.index_select(0, active)
            current_visit = visit_flat.index_select(0, encounter_indices)
            proposed = (
                active_state * decay[:, :, None]
                + torch.sigmoid(self.memory_gate(current_visit))[:, :, None]
                * self.memory_value(current_visit)[:, None, :]
            )
            state = state.index_copy(0, active, proposed)
            previous_days = previous_days.index_copy(0, active, current_days)
            history_indices.append(encounter_indices)
            history_values.append(proposed)

        encounter_order = torch.cat(history_indices)
        memories = visit_flat.new_empty((n_encounters, k, d)).index_copy(
            0, encounter_order, torch.cat(history_values)
        )
        memories = self.memory_norm(
            memories + self.memory_channel_embedding[None, :, :]
        )
        demographic_token = self.demographic_encoder(torch.stack((ages, sexes), dim=-1))
        current_token = self.current_visit_norm(visit_flat)
        keys = torch.cat(
            (memories, current_token[:, None, :], demographic_token[:, None, :]),
            dim=1,
        )
        queries = self.interval_queries[None, :, :].expand(n_encounters, -1, -1)
        readout, attention = self.readout_attention(
            queries,
            keys,
            keys,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        readout = self.dropout(self.readout_norm(readout + queries))
        raw_hazards = (
            torch.einsum("nhd,hcd->nhc", readout, self.hazard_weight)
            + self.hazard_bias[None, :, :]
        )

        diagnosis_count = diagnosis_counts.to(dtype=current_token.dtype)
        if torch.any(diagnosis_count < 0):
            raise ValueError("diagnosis_counts must be nonnegative")
        detached_covariates = torch.stack(
            (
                ages.detach().to(dtype=current_token.dtype),
                sexes.detach().to(dtype=current_token.dtype),
                torch.log1p(diagnosis_count).detach(),
            ),
            dim=-1,
        )
        domain_input = torch.cat(
            (
                gradient_reverse(current_token, domain_coefficient),
                detached_covariates,
            ),
            dim=-1,
        )
        domain_logits = self.domain_classifier(domain_input).squeeze(-1)
        return RadarOutput(
            raw_hazards=raw_hazards,
            domain_logits=domain_logits,
            current_visit_representation=(
                current_token if return_representation else None
            ),
            attention=attention if return_attention else None,
        )
