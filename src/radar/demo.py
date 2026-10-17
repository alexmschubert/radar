"""Generate a tiny synthetic directory with the MIMIC-IV hospital schema."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def build_synthetic_mimic(out_dir: Path, n_subjects: int = 12) -> None:
    if n_subjects < 6:
        raise ValueError("the demo needs at least six subjects")
    hosp = out_dir / "hosp"
    hosp.mkdir(parents=True, exist_ok=False)
    resources = out_dir / "resources"
    resources.mkdir()
    patients: list[dict] = []
    admissions: list[dict] = []
    diagnoses: list[dict] = []
    procedures: list[dict] = []

    for local in range(n_subjects):
        subject_id = 100_000 + local
        base = pd.Timestamp("2014-01-01") + pd.to_timedelta(int(local * 17), unit="D")
        pattern = local % 4
        if pattern == 0:
            offsets = [0, 100, 220]
            af_offset = 220
            death = None
        elif pattern == 1:
            offsets = [0, 60]
            af_offset = None
            death = base + pd.to_timedelta(120, unit="D")
        elif pattern == 2:
            offsets = [0, 180, 400]
            af_offset = None
            death = None
        else:
            offsets = [0, 150, 500]
            af_offset = 500
            death = None
        patients.append(
            {
                "subject_id": subject_id,
                "gender": "M" if local % 2 else "F",
                "anchor_age": 50 + local,
                "anchor_year": base.year,
                "anchor_year_group": "2014 - 2016",
                "dod": death,
            }
        )
        for sequence, offset in enumerate(offsets, start=1):
            hadm_id = subject_id * 10 + sequence
            time = base + pd.to_timedelta(int(offset), unit="D")
            admissions.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "admittime": time,
                    "dischtime": time + pd.to_timedelta(2, unit="D"),
                    "deathtime": None,
                    "admission_type": "EW EMER.",
                    "race": "WHITE" if local % 3 else "BLACK/AFRICAN AMERICAN",
                }
            )
            version = 9 if time < pd.Timestamp("2015-10-01") else 10
            ordinary = (
                ["4019", "4280", "25000", "2724"]
                if version == 9
                else ["I10", "I509", "E119", "E785"]
            )
            for code_index, code in enumerate(
                ordinary[(local + sequence) % len(ordinary) :] + ordinary[:1],
                start=1,
            ):
                diagnoses.append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "seq_num": code_index,
                        "icd_code": code,
                        "icd_version": version,
                    }
                )
            if af_offset == offset:
                diagnoses.append(
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "seq_num": 99,
                        "icd_code": "42731" if version == 9 else "I4891",
                        "icd_version": version,
                    }
                )
                # Exercise both ICD generations of the postoperative AF washout.
                if local in {0, 11}:
                    procedures.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "seq_num": 1,
                            "chartdate": time,
                            "icd_code": "3521" if version == 9 else "02RF0JZ",
                            "icd_version": version,
                        }
                    )

    pd.DataFrame(patients).to_csv(
        hosp / "patients.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(admissions).to_csv(
        hosp / "admissions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(diagnoses).to_csv(
        hosp / "diagnoses_icd.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(
        procedures,
        columns=[
            "subject_id",
            "hadm_id",
            "seq_num",
            "chartdate",
            "icd_code",
            "icd_version",
        ],
    ).to_csv(hosp / "procedures_icd.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        [
            {"icd10cm": "I10", "icd9cm": "4019", "no_map": "0"},
            {"icd10cm": "I509", "icd9cm": "4280", "no_map": "0"},
            {"icd10cm": "E119", "icd9cm": "25000", "no_map": "0"},
            {"icd10cm": "E785", "icd9cm": "2724", "no_map": "0"},
            {"icd10cm": "I4891", "icd9cm": "42731", "no_map": "0"},
        ]
    ).to_csv(resources / "icd10_to_icd9_gem.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} exists; pass --overwrite")
        shutil.rmtree(args.out_dir)
    build_synthetic_mimic(args.out_dir, args.subjects)
    print(f"wrote synthetic MIMIC-shaped files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
