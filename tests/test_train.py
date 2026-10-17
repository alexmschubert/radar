from __future__ import annotations

import pytest
import torch

from radar import train
from radar.demo import build_synthetic_mimic
from radar.mimic import main as prepare_main


def test_validation_auc_pools_batches_and_omits_unobservable_outcomes(monkeypatch):
    batches = [
        {
            "labels_365d": torch.tensor([True, False]),
            "label_masks_365d": torch.tensor([True, False]),
            "scores": torch.tensor([0.9, 0.99]),
        },
        {
            "labels_365d": torch.tensor([False, False, True]),
            "label_masks_365d": torch.tensor([True, True, False]),
            "scores": torch.tensor([0.2, 0.9, 0.01]),
        },
    ]
    monkeypatch.setattr(
        train,
        "batch_loss",
        lambda model, batch, **kwargs: (torch.tensor(1.0), 1.0, 0.0, batch["scores"]),
    )
    metrics = train.evaluate(torch.nn.Module(), batches, torch.device("cpu"), 1.0, 0)
    assert metrics["auroc_365d"] == pytest.approx(0.75)
    with pytest.raises(RuntimeError, match="both cases and controls"):
        train.evaluate(torch.nn.Module(), batches, torch.device("cpu"), 1.0, 1)


def test_checkpoint_selection_uses_auc_even_when_likelihood_improves(
    tmp_path, monkeypatch
):
    mimic = tmp_path / "mimic"
    prepared = tmp_path / "prepared"
    output = tmp_path / "model"
    build_synthetic_mimic(mimic)
    prepare_main(
        [
            "--mimic-dir",
            str(mimic),
            "--gem",
            str(mimic / "resources/icd10_to_icd9_gem.csv"),
            "--out-dir",
            str(prepared),
            "--min-code-count",
            "1",
            "--seed",
            "0",
        ]
    )
    validations = iter(
        [
            {"auroc_365d": 0.9, "clinical_nll": 3.0},
            {"auroc_365d": 0.8, "clinical_nll": 2.0},
            {"auroc_365d": 0.85, "clinical_nll": 1.0},
        ]
    )
    monkeypatch.setattr(train, "evaluate", lambda *args: next(validations))
    train.main(
        [
            "--data-dir",
            str(prepared),
            "--out-dir",
            str(output),
            "--epochs",
            "3",
            "--embedding-dim",
            "8",
            "--batch-patients",
            "1",
            "--device",
            "cpu",
        ]
    )
    best = torch.load(output / "checkpoint_best.pt", weights_only=True)
    last = torch.load(output / "checkpoint_last.pt", weights_only=True)
    assert best["epoch"] == 1
    assert last["epoch"] == 3
    assert best["selection_metric"] == "validation_auroc_365d"
    assert best["validation"]["auroc_365d"] == 0.9
    assert last["scheduler_state"]["_last_lr"] == [0.0]


def test_learning_rate_warmup_peak_and_cosine_decay():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: train.learning_rate_multiplier(step, 100, 0.05)
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.2)
    assert rates[5] == 1.0
    assert rates[5] > rates[50] > rates[99] > rates[100] == 0.0
    assert train.learning_rate_multiplier(0, 100, 0.0) == 1.0
