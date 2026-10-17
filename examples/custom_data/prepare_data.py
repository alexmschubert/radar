#!/usr/bin/env python3
"""Adapt an encounter-level CSV to RADAR's prepared dataset format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from radar.data import SPLIT_IDS, save_dataset


REQUIRED_COLUMNS = {
    "subject_id",
    "encounter_id",
    "split",
    "event_day",
    "age",
    "sex",
    "codes",
    "followup_days",
    "event_type",
    "risk_eligible",
}


def parse_codes(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [token.strip() for token in str(value).split("|") if token.strip()]


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"input CSV is missing columns: {missing}")
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(frame["split"]).difference(SPLIT_IDS))
    if invalid_splits:
        raise ValueError(f"invalid split names: {invalid_splits}")
    if set(frame["split"]) != set(SPLIT_IDS):
        raise ValueError("input must contain train, validation, and test patients")

    integer_columns = [
        "subject_id",
        "encounter_id",
        "sex",
        "event_type",
        "risk_eligible",
    ]
    float_columns = ["event_day", "age", "followup_days"]
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    for column in float_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if "coding_domain" not in frame:
        frame["coding_domain"] = -1
    frame["coding_domain"] = pd.to_numeric(
        frame["coding_domain"], errors="raise"
    ).astype(np.int8)

    if frame["encounter_id"].duplicated().any():
        raise ValueError("encounter_id values must be unique")
    if frame.groupby("subject_id")["split"].nunique().max() != 1:
        raise ValueError("a subject cannot appear in more than one split")
    if not frame["sex"].isin([0, 1]).all():
        raise ValueError("sex must be encoded as female=0 or male=1")
    if not frame["event_type"].isin([0, 1, 2]).all():
        raise ValueError("event_type must be 0, 1, or 2")
    if not frame["risk_eligible"].isin([0, 1]).all():
        raise ValueError("risk_eligible must be 0 or 1")
    if not frame["coding_domain"].isin([-1, 0, 1]).all():
        raise ValueError("coding_domain must be -1, 0, or 1")
    if not np.isfinite(frame[["event_day", "age", "followup_days"]]).all().all():
        raise ValueError("event_day, age, and followup_days must be finite")
    if ((frame["followup_days"] < 0) | (frame["followup_days"] > 365)).any():
        raise ValueError("followup_days must lie between 0 and 365")

    frame["parsed_codes"] = frame["codes"].map(parse_codes)
    return frame.sort_values(
        ["subject_id", "event_day", "encounter_id"], kind="stable"
    ).reset_index(drop=True)


def prepare(
    input_csv: Path,
    out_dir: Path,
    min_code_count: int,
    cause_names: tuple[str, str],
) -> None:
    if min_code_count <= 0:
        raise ValueError("min_code_count must be positive")
    frame = validate_frame(pd.read_csv(input_csv))
    train_rows = frame[frame["split"].eq("train")]
    counts: Counter[str] = Counter()
    for codes in train_rows["parsed_codes"]:
        counts.update(set(codes))
    retained = sorted(
        (token for token, count in counts.items() if count >= min_code_count),
        key=lambda token: (-counts[token], token),
    )
    tokens = ["<UNK>", *retained]
    token_to_id = {token: index for index, token in enumerate(tokens)}

    age_mean = float(train_rows["age"].mean())
    age_std = float(train_rows["age"].std(ddof=0))
    if not np.isfinite(age_std) or age_std <= 0:
        age_std = 1.0

    code_ids: list[int] = []
    code_offsets = [0]
    patient_offsets = [0]
    subject_ids: list[int] = []
    patient_splits: list[int] = []
    for subject_id, patient in frame.groupby("subject_id", sort=False):
        subject_ids.append(int(subject_id))
        patient_splits.append(SPLIT_IDS[str(patient["split"].iloc[0])])
        for codes in patient["parsed_codes"]:
            ids = sorted({token_to_id.get(token, 0) for token in codes})
            code_ids.extend(ids)
            code_offsets.append(len(code_ids))
        patient_offsets.append(patient_offsets[-1] + len(patient))

    arrays = {
        "code_ids": np.asarray(code_ids, dtype=np.int32),
        "code_offsets": np.asarray(code_offsets, dtype=np.int64),
        "patient_offsets": np.asarray(patient_offsets, dtype=np.int64),
        "event_days": frame["event_day"].to_numpy(dtype=np.float32),
        "ages": ((frame["age"] - age_mean) / age_std).to_numpy(dtype=np.float32),
        "sexes": frame["sex"].to_numpy(dtype=np.float32),
        "diagnosis_counts": frame["parsed_codes"].map(len).to_numpy(dtype=np.float32),
        "coding_domains": frame["coding_domain"].to_numpy(dtype=np.float32),
        "followup_days": frame["followup_days"].to_numpy(dtype=np.float32),
        "event_types": frame["event_type"].to_numpy(dtype=np.uint8),
        "risk_masks": frame["risk_eligible"].to_numpy(dtype=np.bool_),
        "subject_ids": np.asarray(subject_ids, dtype=np.int64),
        "hadm_ids": frame["encounter_id"].to_numpy(dtype=np.int64),
        "patient_splits": np.asarray(patient_splits, dtype=np.uint8),
    }
    split_counts = {split: int((frame["split"] == split).sum()) for split in SPLIT_IDS}
    metadata = {
        "format_version": 1,
        "source": "custom encounter CSV",
        "source_file": str(input_csv.resolve()),
        "cause_names": list(cause_names),
        "patients": len(subject_ids),
        "encounters": len(frame),
        "vocab_size": len(tokens),
        "encounter_split_counts": split_counts,
        "age_mean_training": age_mean,
        "age_std_training": age_std,
        "event_types": {
            "0": "censored/event-free",
            "1": cause_names[0],
            "2": cause_names[1],
        },
        "coding_domain": "-1 ignored; 0 domain A; 1 domain B",
    }
    vocabulary = {
        "format_version": 1,
        "unknown_token": "<UNK>",
        "token_to_id": token_to_id,
        "id_to_token": tokens,
        "counts_in_training_patients": {
            token: int(counts[token]) for token in retained
        },
    }
    save_dataset(out_dir, arrays, metadata, vocabulary)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-code-count", type=int, default=1)
    parser.add_argument("--cause-1-name", default="cause_1")
    parser.add_argument("--cause-2-name", default="cause_2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        args.input,
        args.out_dir,
        args.min_code_count,
        (args.cause_1_name, args.cause_2_name),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
