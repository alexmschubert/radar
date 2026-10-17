"""Memory-mapped patient sequence data used by RADAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
SPLIT_IDS = {value: key for key, value in SPLIT_NAMES.items()}


class RadarDataset(Dataset):
    """A dataset whose items are complete patient trajectories."""

    ARRAY_NAMES = (
        "code_ids",
        "code_offsets",
        "patient_offsets",
        "event_days",
        "ages",
        "sexes",
        "diagnosis_counts",
        "coding_domains",
        "followup_days",
        "event_types",
        "risk_masks",
        "subject_ids",
        "hadm_ids",
        "patient_splits",
    )

    def __init__(self, directory: Path | str, split: str) -> None:
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.vocabulary = json.loads((self.directory / "vocabulary.json").read_text())
        for name in self.ARRAY_NAMES:
            setattr(
                self,
                name,
                np.load(self.directory / f"{name}.npy", mmap_mode="r"),
            )
        label_paths = [
            self.directory / f"{name}.npy"
            for name in ("labels_365d", "label_masks_365d")
        ]
        if any(path.exists() for path in label_paths):
            if not all(path.exists() for path in label_paths):
                raise ValueError("12-month labels and masks must be provided together")
            self.labels_365d, self.label_masks_365d = (
                np.load(path, mmap_mode="r") for path in label_paths
            )
        else:
            self.labels_365d = self.label_masks_365d = None
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {sorted(SPLIT_IDS)}")
        self.patient_indices = np.flatnonzero(self.patient_splits == SPLIT_IDS[split])
        if not len(self.patient_indices):
            raise ValueError(f"dataset has no patients in split {split!r}")

    def __len__(self) -> int:
        return int(len(self.patient_indices))

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        patient_index = int(self.patient_indices[index])
        encounter_start = int(self.patient_offsets[patient_index])
        encounter_end = int(self.patient_offsets[patient_index + 1])
        code_start = int(self.code_offsets[encounter_start])
        code_end = int(self.code_offsets[encounter_end])
        encounter_slice = slice(encounter_start, encounter_end)
        # Derive binary labels from survival targets when explicit arrays
        # are absent; include events and fully observed controls.
        labels = (
            np.asarray(self.event_types[encounter_slice]) == 1
            if self.labels_365d is None
            else np.asarray(self.labels_365d[encounter_slice])
        )
        label_masks = (
            np.asarray(self.risk_masks[encounter_slice])
            & (labels | (np.asarray(self.followup_days[encounter_slice]) >= 365))
            if self.label_masks_365d is None
            else np.asarray(self.label_masks_365d[encounter_slice])
        )
        return {
            "code_ids": np.asarray(self.code_ids[code_start:code_end]),
            "code_offsets": np.asarray(
                self.code_offsets[encounter_start : encounter_end + 1] - code_start
            ),
            "event_days": np.asarray(self.event_days[encounter_slice]),
            "ages": np.asarray(self.ages[encounter_slice]),
            "sexes": np.asarray(self.sexes[encounter_slice]),
            "diagnosis_counts": np.asarray(self.diagnosis_counts[encounter_slice]),
            "coding_domains": np.asarray(self.coding_domains[encounter_slice]),
            "followup_days": np.asarray(self.followup_days[encounter_slice]),
            "event_types": np.asarray(self.event_types[encounter_slice]),
            "risk_masks": np.asarray(self.risk_masks[encounter_slice]),
            "labels_365d": labels,
            "label_masks_365d": label_masks,
            "hadm_ids": np.asarray(self.hadm_ids[encounter_slice]),
            "subject_id": int(self.subject_ids[patient_index]),
        }


def collate_patient_sequences(items: Sequence[dict]) -> dict[str, torch.Tensor]:
    """Combine complete patient trajectories into one flat CSR batch."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    code_ids: list[np.ndarray] = []
    code_offsets = [0]
    patient_ptr = [0]
    encounter_arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "event_days",
            "ages",
            "sexes",
            "diagnosis_counts",
            "coding_domains",
            "followup_days",
            "event_types",
            "risk_masks",
            "labels_365d",
            "label_masks_365d",
            "hadm_ids",
        )
    }
    subject_ids: list[int] = []
    encounter_subject_ids: list[np.ndarray] = []
    code_count = 0
    encounter_count = 0

    for item in items:
        local_codes = np.asarray(item["code_ids"], dtype=np.int64)
        local_offsets = np.asarray(item["code_offsets"], dtype=np.int64)
        n_encounters = len(local_offsets) - 1
        code_ids.append(local_codes)
        code_offsets.extend((local_offsets[1:] + code_count).tolist())
        code_count += len(local_codes)
        encounter_count += n_encounters
        patient_ptr.append(encounter_count)
        for name in encounter_arrays:
            encounter_arrays[name].append(np.asarray(item[name]))
        subject_id = int(item["subject_id"])
        subject_ids.append(subject_id)
        encounter_subject_ids.append(np.full(n_encounters, subject_id, dtype=np.int64))

    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(encounter_arrays[name]), dtype=dtype)

    return {
        "code_ids": torch.as_tensor(np.concatenate(code_ids), dtype=torch.long),
        "code_offsets": torch.as_tensor(code_offsets, dtype=torch.long),
        "patient_ptr": torch.as_tensor(patient_ptr, dtype=torch.long),
        "event_days": tensor("event_days", torch.float32),
        "ages": tensor("ages", torch.float32),
        "sexes": tensor("sexes", torch.float32),
        "diagnosis_counts": tensor("diagnosis_counts", torch.float32),
        "coding_domains": tensor("coding_domains", torch.float32),
        "followup_days": tensor("followup_days", torch.float32),
        "event_types": tensor("event_types", torch.long),
        "risk_masks": tensor("risk_masks", torch.bool),
        "labels_365d": tensor("labels_365d", torch.bool),
        "label_masks_365d": tensor("label_masks_365d", torch.bool),
        "hadm_ids": tensor("hadm_ids", torch.long),
        "subject_ids": torch.as_tensor(subject_ids, dtype=torch.long),
        "encounter_subject_ids": torch.as_tensor(
            np.concatenate(encounter_subject_ids), dtype=torch.long
        ),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


def save_dataset(
    directory: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    vocabulary: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", np.asarray(values), allow_pickle=False)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (directory / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True) + "\n"
    )
