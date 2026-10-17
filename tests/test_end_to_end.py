from __future__ import annotations

import csv
import json

from radar.demo import main as demo_main
from radar.mimic import main as mimic_main
from radar.predict import main as predict_main
from radar.train import main as train_main


def test_synthetic_mimic_training_and_prediction(tmp_path):
    mimic = tmp_path / "mimic"
    prepared = tmp_path / "prepared"
    model = tmp_path / "model"
    predictions = tmp_path / "predictions"
    assert demo_main(["--out-dir", str(mimic)]) == 0
    assert (
        mimic_main(
            [
                "--mimic-dir",
                str(mimic),
                "--gem",
                str(mimic / "resources" / "icd10_to_icd9_gem.csv"),
                "--out-dir",
                str(prepared),
                "--min-code-count",
                "1",
                "--seed",
                "0",
                "--csv-chunk-rows",
                "17",
            ]
        )
        == 0
    )
    metadata = json.loads((prepared / "metadata.json").read_text())
    assert metadata["patients"] == 12
    assert metadata["vocab_size"] > 1
    assert metadata["risk_eligible_encounters"] > 0
    assert metadata["open_heart_surgery_patients"] == 2
    assert (
        metadata["af_outcome_patients_after_surgery_exclusion"]
        < metadata["af_outcome_admissions_before_surgery_exclusion"]
    )
    vocabulary = json.loads((prepared / "vocabulary.json").read_text())
    assert all(
        token == "<UNK>" or token.startswith("ICD9_")
        for token in vocabulary["id_to_token"]
    )
    assert "ICD9_42731" not in vocabulary["id_to_token"]
    assert "ICD9_42732" not in vocabulary["id_to_token"]
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
                "4",
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
    assert (model / "checkpoint_best.pt").is_file()
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
    manifest = json.loads((predictions / "manifest.json").read_text())
    assert manifest["probability_invariants_verified"]
    with (predictions / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == metadata["encounter_split_counts"]["test"]
