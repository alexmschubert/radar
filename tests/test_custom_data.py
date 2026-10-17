from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from radar.data import RadarDataset
from radar.predict import main as predict_main
from radar.train import main as train_main


REPOSITORY = Path(__file__).resolve().parents[1]
CUSTOM_MAIN = runpy.run_path(str(REPOSITORY / "examples/custom_data/prepare_data.py"))[
    "main"
]


def test_custom_adapter_with_script_first_training_and_prediction(tmp_path):
    prepared = tmp_path / "prepared"
    model = tmp_path / "model"
    predictions = tmp_path / "predictions"
    assert (
        CUSTOM_MAIN(
            [
                "--input",
                str(REPOSITORY / "examples/custom_data/example_encounters.csv"),
                "--out-dir",
                str(prepared),
                "--cause-1-name",
                "outcome",
                "--cause-2-name",
                "death",
            ]
        )
        == 0
    )
    metadata = json.loads((prepared / "metadata.json").read_text())
    assert metadata["patients"] == 9
    assert metadata["cause_names"] == ["outcome", "death"]
    assert all(
        len(RadarDataset(prepared, split)) > 0
        for split in (
            "train",
            "validation",
            "test",
        )
    )

    assert (
        train_main(
            [
                "--data-dir",
                str(prepared),
                "--out-dir",
                str(model),
                "--epochs",
                "1",
                "--batch-patients",
                "2",
                "--embedding-dim",
                "16",
                "--memory-channels",
                "4",
                "--attention-heads",
                "4",
                "--domain-hidden-dim",
                "8",
                "--dropout",
                "0",
                "--half-lives-days",
                "7",
                "30",
                "365",
                "36500",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (
        predict_main(
            [
                "--checkpoint",
                str(model / "checkpoint_best.pt"),
                "--data-dir",
                str(prepared),
                "--out-dir",
                str(predictions),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    with (predictions / "predictions.csv").open() as handle:
        header = next(csv.reader(handle))
    assert "outcome_90d" in header
    assert "outcome_365d" in header
    assert "death_365d" in header
