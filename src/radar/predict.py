"""Run a trained RADAR checkpoint on a prepared dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import RadarDataset, collate_patient_sequences, move_batch
from .model import RadarConfig, RadarModel, cumulative_incidence
from .train import resolve_device


def output_name(value: object) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not name:
        raise ValueError("cause names must contain a letter or number")
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-patients", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} already exists")
    args.out_dir.mkdir(parents=True)
    device = resolve_device(args.device)
    dataset = RadarDataset(args.data_dir, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config_values = dict(checkpoint["model_config"])
    config_values["half_lives_days"] = tuple(config_values["half_lives_days"])
    config = RadarConfig(**config_values)
    if config.vocab_size != int(dataset.metadata["vocab_size"]):
        raise RuntimeError("checkpoint and dataset vocabulary sizes differ")
    model = RadarModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    cause_names = dataset.metadata.get("cause_names", ["af", "death"])
    if not isinstance(cause_names, list) or len(cause_names) != 2:
        raise RuntimeError("dataset metadata cause_names must contain two names")
    cause_1, cause_2 = map(output_name, cause_names)
    if cause_1 == cause_2:
        raise RuntimeError("dataset cause names must be distinct")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_patients,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patient_sequences,
    )
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                code_ids=batch["code_ids"],
                code_offsets=batch["code_offsets"],
                patient_ptr=batch["patient_ptr"],
                event_days=batch["event_days"],
                ages=batch["ages"],
                sexes=batch["sexes"],
                diagnosis_counts=batch["diagnosis_counts"],
            )
            first_cif, second_cif, survival = cumulative_incidence(output.raw_hazards)
            arrays = {
                "subject_id": batch["encounter_subject_ids"].cpu().numpy(),
                "hadm_id": batch["hadm_ids"].cpu().numpy(),
                "followup_days": batch["followup_days"].cpu().numpy(),
                "event_type": batch["event_types"].cpu().numpy(),
                "risk_eligible": batch["risk_masks"].cpu().numpy(),
                f"{cause_1}_90d": first_cif[:, 2].cpu().numpy(),
                f"{cause_1}_180d": first_cif[:, 5].cpu().numpy(),
                f"{cause_1}_365d": first_cif[:, 11].cpu().numpy(),
                f"{cause_2}_365d": second_cif[:, 11].cpu().numpy(),
                "event_free_365d": survival[:, 11].cpu().numpy(),
            }
            for index in range(len(arrays["hadm_id"])):
                rows.append(
                    {
                        name: int(values[index])
                        if name
                        in {
                            "subject_id",
                            "hadm_id",
                            "event_type",
                            "risk_eligible",
                        }
                        else float(values[index])
                        for name, values in arrays.items()
                    }
                )

    output_path = args.out_dir / "predictions.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "rows": len(rows),
        "device": str(device),
        "probability_invariants_verified": bool(
            all(
                np.isclose(
                    row[f"{cause_1}_365d"]
                    + row[f"{cause_2}_365d"]
                    + row["event_free_365d"],
                    1.0,
                    atol=2e-5,
                )
                for row in rows
            )
        ),
        "cause_names": [cause_1, cause_2],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(rows):,} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
