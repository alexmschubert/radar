from __future__ import annotations

import torch

from radar.losses import competing_risk_nll
from radar.model import RadarConfig, RadarModel, cumulative_incidence


def tiny_model() -> RadarModel:
    torch.manual_seed(7)
    return RadarModel(
        RadarConfig(
            vocab_size=12,
            embedding_dim=16,
            visit_queries=2,
            memory_channels=4,
            attention_heads=4,
            domain_hidden_dim=8,
            dropout=0.0,
            half_lives_days=(7, 30, 365, 36500),
        )
    )


def test_forward_probability_invariants_and_gradients():
    model = tiny_model()
    output = model(
        code_ids=torch.tensor([1, 2, 3, 4, 5]),
        code_offsets=torch.tensor([0, 2, 3, 5]),
        patient_ptr=torch.tensor([0, 2, 3]),
        event_days=torch.tensor([0.0, 100.0, 20.0]),
        ages=torch.tensor([0.0, 0.1, -0.2]),
        sexes=torch.tensor([0.0, 0.0, 1.0]),
        diagnosis_counts=torch.tensor([2.0, 1.0, 2.0]),
        domain_coefficient=0.01,
    )
    af, death, survival = cumulative_incidence(output.raw_hazards)
    assert af.shape == death.shape == survival.shape == (3, 12)
    assert torch.all(torch.diff(af, dim=1) >= -1e-7)
    assert torch.all(torch.diff(death, dim=1) >= -1e-7)
    torch.testing.assert_close(
        af + death + survival, torch.ones_like(af), rtol=2e-5, atol=2e-6
    )
    loss = competing_risk_nll(
        output.raw_hazards,
        torch.tensor([100.0, 365.0, 40.0]),
        torch.tensor([1, 0, 2]),
        torch.ones(3, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_diagnosis_order_does_not_change_predictions():
    model = tiny_model().eval()
    shared = {
        "code_offsets": torch.tensor([0, 3]),
        "patient_ptr": torch.tensor([0, 1]),
        "event_days": torch.tensor([0.0]),
        "ages": torch.tensor([0.0]),
        "sexes": torch.tensor([1.0]),
        "diagnosis_counts": torch.tensor([3.0]),
    }
    first = model(code_ids=torch.tensor([1, 2, 3]), **shared)
    second = model(code_ids=torch.tensor([3, 1, 2]), **shared)
    torch.testing.assert_close(first.raw_hazards, second.raw_hazards)


def test_predictions_do_not_use_future_encounters_or_other_patients():
    model = tiny_model().eval()
    inputs = {
        "code_ids": torch.tensor([1, 2, 3, 4]),
        "code_offsets": torch.tensor([0, 1, 2, 3, 4]),
        "patient_ptr": torch.tensor([0, 3, 4]),
        "event_days": torch.tensor([0.0, 30.0, 90.0, 0.0]),
        "ages": torch.zeros(4),
        "sexes": torch.zeros(4),
        "diagnosis_counts": torch.ones(4),
    }
    original = model(**inputs).raw_hazards
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["code_ids"][2:] = torch.tensor([10, 11])
    changed["event_days"][2] = 180.0
    changed["ages"][2:] = 2.0
    torch.testing.assert_close(original[:2], model(**changed).raw_hazards[:2])


def test_coding_adversary_covariates_do_not_change_clinical_predictions():
    model = tiny_model().eval()
    inputs = {
        "code_ids": torch.tensor([1, 2]),
        "code_offsets": torch.tensor([0, 2]),
        "patient_ptr": torch.tensor([0, 1]),
        "event_days": torch.tensor([0.0]),
        "ages": torch.tensor([0.0]),
        "sexes": torch.tensor([1.0]),
    }
    original = model(**inputs, diagnosis_counts=torch.tensor([2.0]))
    changed = model(**inputs, diagnosis_counts=torch.tensor([200.0]))
    torch.testing.assert_close(original.raw_hazards, changed.raw_hazards)
