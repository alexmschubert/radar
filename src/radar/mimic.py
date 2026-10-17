"""Prepare MIMIC-IV v3.x hospital data for training RADAR.

Expected files are ``hosp/patients.csv.gz``, ``hosp/admissions.csv.gz``,
``hosp/diagnoses_icd.csv.gz``, and (for the postoperative washout)
``hosp/procedures_icd.csv.gz``. Uncompressed ``.csv`` files also work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import save_dataset


AF_ICD9 = frozenset({"42731", "42732"})
AF_ICD10_PREFIXES = ("I48",)
OHS_ICD9_PREFIXES = (
    "3510",
    "3511",
    "3512",
    "3513",
    "3514",
    "3520",
    "3521",
    "3522",
    "3523",
    "3524",
    "3525",
    "3526",
    "3527",
    "3528",
    "3531",
    "3532",
    "3533",
    "3534",
    "3535",
    "3539",
    "3541",
    "3542",
    "3550",
    "3551",
    "3552",
    "3553",
    "3554",
    "3560",
    "3561",
    "3562",
    "3563",
    "3570",
    "3571",
    "3572",
    "3573",
    "3603",
    "3610",
    "3611",
    "3612",
    "3613",
    "3614",
    "3615",
    "3616",
    "3617",
    "3618",
    "3619",
    "3631",
    "3710",
    "3711",
    "3712",
    "3731",
    "3732",
    "3733",
    "3735",
    "3741",
)
OHS_ICD10_PREFIXES = (
    "02RF",
    "02RG",
    "02RH",
    "02RJ",
    "02QF",
    "02QG",
    "02QH",
    "02QJ",
    "02UF",
    "02UG",
    "02UH",
    "02UJ",
    "0210",
    "0211",
    "0212",
    "0213",
    "02YA0Z0",
    "02QA",
    "02QB",
    "02QC",
    "02QD",
    "02HA",
    "02HK",
    "02HL",
    "02HM",
    "02HN",
)


def clean_code(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "".join(character for character in str(value).upper() if character.isalnum())


def is_af(code: str, version: int) -> bool:
    return code in AF_ICD9 if version == 9 else code.startswith(AF_ICD10_PREFIXES)


def is_ohs(code: str, version: int) -> bool:
    prefixes = OHS_ICD9_PREFIXES if version == 9 else OHS_ICD10_PREFIXES
    return any(code.startswith(prefix) for prefix in prefixes)


def source_file(mimic_dir: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = mimic_dir / "hosp" / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing hosp/{stem}.csv.gz (or .csv) below {mimic_dir}")


def _add_gem_mapping(mapping: dict[str, str], source: object, target: object) -> None:
    source_code = clean_code(source)
    target_code = clean_code(target)
    if source_code and target_code:
        # Retain the first listed GEM target, including the NoDx status token.
        mapping.setdefault(source_code, target_code)


def load_gem(path: Path) -> tuple[dict[str, str], str]:
    """Load a forward ICD-10-CM -> ICD-9-CM GEM from CMS text/ZIP or CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"GEM file does not exist: {path}")
    mapping: dict[str, str] = {}
    member_name = path.name

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = sorted(
                name
                for name in archive.namelist()
                if Path(name).name.lower().endswith("_i10gem.txt")
            )
            if not candidates:
                raise RuntimeError(
                    f"{path} does not contain a forward *_I10gem.txt file"
                )
            member_name = candidates[-1]
            lines = archive.read(member_name).decode("utf-8-sig").splitlines()
        for line in lines:
            fields = line.split()
            if len(fields) < 2:
                continue
            _add_gem_mapping(mapping, fields[0], fields[1])
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        columns = {column.lower().strip(): column for column in frame.columns}
        source_column = columns.get("icd10cm") or columns.get("icd10")
        target_column = columns.get("icd9cm") or columns.get("icd9")
        if source_column is None or target_column is None:
            raise RuntimeError("GEM CSV needs icd10cm/icd9cm (or icd10/icd9) columns")
        for row in frame.to_dict(orient="records"):
            _add_gem_mapping(mapping, row[source_column], row[target_column])
    else:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            _add_gem_mapping(mapping, fields[0], fields[1])

    if not mapping:
        raise RuntimeError(f"no usable ICD-10-to-ICD-9 mappings found in {path}")
    return mapping, member_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_splits(subject_ids: np.ndarray, seed: int) -> dict[int, int]:
    """Create deterministic 70/15/15 patient-level train/validation/test splits."""
    unique = np.asarray(sorted(set(map(int, subject_ids))), dtype=np.int64)
    if len(unique) < 3:
        raise ValueError("at least three MIMIC subjects are required")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, round(0.15 * len(shuffled)))
    n_validation = max(1, round(0.15 * len(shuffled)))
    if n_test + n_validation >= len(shuffled):
        n_test = n_validation = 1
    mapping: dict[int, int] = {}
    for value in shuffled[:n_test]:
        mapping[int(value)] = 2
    for value in shuffled[n_test : n_test + n_validation]:
        mapping[int(value)] = 1
    for value in shuffled[n_test + n_validation :]:
        mapping[int(value)] = 0
    return mapping


def in_washout(
    time: pd.Timestamp,
    surgery_dates: tuple[pd.Timestamp, ...],
    days: int,
) -> bool:
    if days <= 0:
        return False
    return any(
        surgery <= time <= surgery + pd.to_timedelta(int(days), unit="D")
        for surgery in surgery_dates
    )


def load_admissions(
    mimic_dir: Path,
    max_subjects: int,
) -> tuple[pd.DataFrame, dict[int, pd.Timestamp | None]]:
    patients = pd.read_csv(
        source_file(mimic_dir, "patients"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    )
    admissions = pd.read_csv(
        source_file(mimic_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    if admissions["admittime"].isna().any():
        raise RuntimeError("one or more admissions have an invalid admittime")
    if admissions["hadm_id"].duplicated().any():
        raise RuntimeError("hadm_id must be unique in admissions")
    if max_subjects:
        selected = np.sort(admissions["subject_id"].unique())[:max_subjects]
        admissions = admissions[admissions["subject_id"].isin(selected)].copy()
        patients = patients[patients["subject_id"].isin(selected)].copy()
    admissions = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    if admissions[["gender", "anchor_age", "anchor_year"]].isna().any().any():
        raise RuntimeError("admissions could not be linked to patient demographics")
    admissions["age"] = (
        admissions["anchor_age"].astype(float)
        + admissions["admittime"].dt.year.astype(float)
        - admissions["anchor_year"].astype(float)
    )
    gender = admissions["gender"].astype(str).str.upper()
    if not gender.isin(["F", "M"]).all():
        raise RuntimeError("MIMIC gender must be F or M")
    admissions["sex_male"] = gender.eq("M")
    admissions = admissions.sort_values(
        ["subject_id", "admittime", "hadm_id"], kind="stable"
    ).reset_index(drop=True)
    deaths = {
        int(row.subject_id): (None if pd.isna(row.dod) else pd.Timestamp(row.dod))
        for row in patients[["subject_id", "dod"]].itertuples(index=False)
    }
    return admissions, deaths


def read_diagnoses(
    path: Path,
    valid_hadm_ids: set[int],
    hadm_splits: dict[int, int],
    gem: dict[str, str],
    chunk_rows: int,
) -> tuple[
    dict[int, set[str]],
    set[int],
    dict[int, tuple[int, int]],
    Counter,
    Counter,
]:
    codes_by_hadm: dict[int, set[str]] = defaultdict(set)
    af_hadm_ids: set[int] = set()
    domain_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    train_counts: Counter[str] = Counter()
    audit: Counter[str] = Counter()
    columns = ["hadm_id", "icd_code", "icd_version"]
    for chunk in pd.read_csv(
        path, usecols=columns, dtype={"icd_code": "string"}, chunksize=chunk_rows
    ):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)]
        for row in chunk.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            version = int(row.icd_version)
            if version not in (9, 10):
                raise RuntimeError(f"unexpected ICD version {version}")
            code = clean_code(row.icd_code)
            if not code:
                continue
            domain_counts[hadm_id][version == 10] += 1
            audit[f"raw_icd{version}_rows"] += 1
            if is_af(code, version):
                af_hadm_ids.add(hadm_id)
                audit["outcome_codes_removed_before_mapping"] += 1
                continue
            if version == 10:
                mapped = gem.get(code)
                if mapped is None:
                    audit["icd10_rows_unmapped_and_omitted"] += 1
                    continue
                code = mapped
                audit["icd10_rows_mapped"] += 1
            else:
                audit["icd9_rows_retained"] += 1
            if code in AF_ICD9 or code.startswith(AF_ICD10_PREFIXES):
                audit["outcome_codes_removed_after_mapping"] += 1
                continue
            token = f"ICD9_{code}"
            codes_by_hadm[hadm_id].add(token)
    for hadm_id, tokens in codes_by_hadm.items():
        if hadm_splits[hadm_id] == 0:
            train_counts.update(tokens)
    frozen_domains = {
        hadm_id: (values[0], values[1]) for hadm_id, values in domain_counts.items()
    }
    return codes_by_hadm, af_hadm_ids, frozen_domains, train_counts, audit


def read_surgeries(
    path: Path,
    valid_hadm_ids: set[int],
    admission_lookup: dict[int, tuple[int, pd.Timestamp]],
    chunk_rows: int,
) -> dict[int, tuple[pd.Timestamp, ...]]:
    by_subject: dict[int, set[pd.Timestamp]] = defaultdict(set)
    columns = ["hadm_id", "chartdate", "icd_code", "icd_version"]
    for chunk in pd.read_csv(
        path, usecols=columns, dtype={"icd_code": "string"}, chunksize=chunk_rows
    ):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)]
        for row in chunk.itertuples(index=False):
            version = int(row.icd_version)
            code = clean_code(row.icd_code)
            if version not in (9, 10) or not code or not is_ohs(code, version):
                continue
            subject_id, admission_time = admission_lookup[int(row.hadm_id)]
            chartdate = pd.to_datetime(row.chartdate, errors="coerce")
            by_subject[subject_id].add(
                admission_time if pd.isna(chartdate) else pd.Timestamp(chartdate)
            )
    return {
        subject_id: tuple(sorted(values)) for subject_id, values in by_subject.items()
    }


def build_vocabulary(
    train_counts: Counter,
    min_code_count: int,
    max_vocab_size: int,
) -> tuple[dict[str, int], dict]:
    eligible = [
        (token, count)
        for token, count in train_counts.items()
        if count >= min_code_count
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab_size:
        eligible = eligible[: max(0, max_vocab_size - 1)]
    tokens = ["<UNK>", *(token for token, _count in eligible)]
    token_to_id = {token: index for index, token in enumerate(tokens)}
    vocabulary = {
        "format_version": 1,
        "unknown_token": "<UNK>",
        "token_to_id": token_to_id,
        "id_to_token": tokens,
        "counts_in_training_patients": {
            token: int(train_counts[token]) for token in tokens[1:]
        },
    }
    return token_to_id, vocabulary


def first_eligible_af_times(
    admissions: pd.DataFrame,
    af_hadm_ids: set[int],
    surgeries: dict[int, tuple[pd.Timestamp, ...]],
    washout_days: int,
) -> dict[int, pd.Timestamp]:
    first: dict[int, pd.Timestamp] = {}
    for row in admissions[admissions["hadm_id"].isin(af_hadm_ids)].itertuples(
        index=False
    ):
        subject_id = int(row.subject_id)
        time = pd.Timestamp(row.admittime)
        if in_washout(time, surgeries.get(subject_id, ()), washout_days):
            continue
        if subject_id not in first or time < first[subject_id]:
            first[subject_id] = time
    return first


def continuous_competing_target(
    *,
    index_day: int,
    last_encounter_day: int,
    af_day: int | None,
    death_day: int | None,
    risk_eligible: bool,
) -> tuple[float, int, bool]:
    """Construct an exact-exposure competing-risk training target.

    A later event establishes observation through its date. AF wins a same-day
    tie with death. Encounters outside the incident risk set have zero exposure.
    """
    if not risk_eligible:
        return 0.0, 0, False
    candidates = []
    for event_day, cause in ((af_day, 1), (death_day, 2)):
        if event_day is not None and event_day > index_day:
            candidates.append((event_day - index_day, cause))
    first_event = min(candidates, default=None)
    followup = max(0, last_encounter_day - index_day)
    if first_event is not None:
        followup = max(followup, first_event[0])
        if first_event[0] <= 365:
            return float(first_event[0]), first_event[1], True
    return float(min(followup, 365)), 0, True


def prepare_arrays(
    admissions: pd.DataFrame,
    deaths: dict[int, pd.Timestamp | None],
    splits: dict[int, int],
    codes_by_hadm: dict[int, set[str]],
    domain_counts: dict[int, tuple[int, int]],
    surgeries: dict[int, tuple[pd.Timestamp, ...]],
    first_af: dict[int, pd.Timestamp],
    token_to_id: dict[str, int],
    washout_days: int,
) -> tuple[dict[str, np.ndarray], dict]:
    train_rows = admissions[admissions["subject_id"].map(splits).eq(0)]
    age_mean = float(train_rows["age"].mean())
    age_std = float(train_rows["age"].std(ddof=0))
    if not np.isfinite(age_std) or age_std <= 0:
        age_std = 1.0

    code_ids: list[int] = []
    code_offsets = [0]
    patient_offsets = [0]
    event_days: list[float] = []
    ages: list[float] = []
    sexes: list[float] = []
    diagnosis_counts: list[float] = []
    coding_domains: list[float] = []
    followup_days: list[float] = []
    event_types: list[int] = []
    risk_masks: list[bool] = []
    labels_365d: list[bool] = []
    label_masks_365d: list[bool] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    patient_splits: list[int] = []

    for subject_id, frame in admissions.groupby("subject_id", sort=False):
        subject_id = int(subject_id)
        frame = frame.sort_values(["admittime", "hadm_id"], kind="stable")
        subject_ids.append(subject_id)
        patient_splits.append(splits[subject_id])
        last_observed = pd.Timestamp(frame["admittime"].max())
        death = deaths.get(subject_id)
        af_time = first_af.get(subject_id)
        subject_surgeries = surgeries.get(subject_id, ())

        for row in frame.itertuples(index=False):
            index_time = pd.Timestamp(row.admittime)
            hadm_id = int(row.hadm_id)
            tokens = codes_by_hadm.get(hadm_id, set())
            ids = sorted({token_to_id.get(token, 0) for token in tokens})
            code_ids.extend(ids)
            code_offsets.append(len(code_ids))
            hadm_ids.append(hadm_id)
            index_day = int(index_time.value // 86_400_000_000_000)
            event_days.append(float(index_day))
            ages.append((float(row.age) - age_mean) / age_std)
            sexes.append(float(bool(row.sex_male)))
            n_icd9, n_icd10 = domain_counts.get(hadm_id, (0, 0))
            diagnosis_counts.append(float(n_icd9 + n_icd10))
            coding_domains.append(
                0.0 if n_icd9 > n_icd10 else 1.0 if n_icd10 > n_icd9 else -1.0
            )

            eligible = (
                af_time is None or (af_time - index_time).days > 0
            ) and not in_washout(index_time, subject_surgeries, washout_days)
            af_within_year = (
                af_time is not None and 0 < (af_time - index_time).days <= 365
            )
            labels_365d.append(af_within_year)
            label_masks_365d.append(
                eligible
                and (af_within_year or (last_observed - index_time).days >= 365)
            )
            exposure, event_type, risk_mask = continuous_competing_target(
                index_day=index_day,
                last_encounter_day=int(last_observed.value // 86_400_000_000_000),
                af_day=None
                if af_time is None
                else int(af_time.value // 86_400_000_000_000),
                death_day=None
                if death is None
                else int(death.value // 86_400_000_000_000),
                risk_eligible=eligible,
            )
            followup_days.append(exposure)
            event_types.append(event_type)
            risk_masks.append(risk_mask)
        patient_offsets.append(len(hadm_ids))

    arrays = {
        "code_ids": np.asarray(code_ids, dtype=np.int32),
        "code_offsets": np.asarray(code_offsets, dtype=np.int64),
        "patient_offsets": np.asarray(patient_offsets, dtype=np.int64),
        "event_days": np.asarray(event_days, dtype=np.float32),
        "ages": np.asarray(ages, dtype=np.float32),
        "sexes": np.asarray(sexes, dtype=np.float32),
        "diagnosis_counts": np.asarray(diagnosis_counts, dtype=np.float32),
        "coding_domains": np.asarray(coding_domains, dtype=np.float32),
        "followup_days": np.asarray(followup_days, dtype=np.float32),
        "event_types": np.asarray(event_types, dtype=np.uint8),
        "risk_masks": np.asarray(risk_masks, dtype=np.bool_),
        "labels_365d": np.asarray(labels_365d, dtype=np.bool_),
        "label_masks_365d": np.asarray(label_masks_365d, dtype=np.bool_),
        "subject_ids": np.asarray(subject_ids, dtype=np.int64),
        "hadm_ids": np.asarray(hadm_ids, dtype=np.int64),
        "patient_splits": np.asarray(patient_splits, dtype=np.uint8),
    }
    facts = {"age_mean_training": age_mean, "age_std_training": age_std}
    return arrays, facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument(
        "--gem",
        type=Path,
        required=True,
        help="CMS forward ICD-10-CM to ICD-9-CM GEM ZIP, text file, or CSV",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-code-count", type=int, default=5)
    parser.add_argument("--max-vocab-size", type=int, default=0)
    parser.add_argument("--ohs-washout-days", type=int, default=90)
    parser.add_argument("--csv-chunk-rows", type=int, default=500_000)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_code_count <= 0 or args.csv_chunk_rows <= 0:
        raise ValueError("counts and chunk size must be positive")
    if args.ohs_washout_days < 0:
        raise ValueError("ohs washout cannot be negative")
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} exists; pass --overwrite")
        shutil.rmtree(args.out_dir)

    admissions, deaths = load_admissions(args.mimic_dir, args.max_subjects)
    gem, gem_member = load_gem(args.gem)
    splits = assign_splits(admissions["subject_id"].to_numpy(), args.seed)
    admissions["split"] = admissions["subject_id"].map(splits)
    valid_hadm_ids = set(map(int, admissions["hadm_id"]))
    hadm_splits = dict(zip(map(int, admissions["hadm_id"]), admissions["split"]))
    codes_by_hadm, af_hadm_ids, domains, train_counts, mapping_audit = read_diagnoses(
        source_file(args.mimic_dir, "diagnoses_icd"),
        valid_hadm_ids,
        hadm_splits,
        gem,
        args.csv_chunk_rows,
    )
    admission_lookup = {
        int(row.hadm_id): (int(row.subject_id), pd.Timestamp(row.admittime))
        for row in admissions.itertuples(index=False)
    }
    surgeries = read_surgeries(
        source_file(args.mimic_dir, "procedures_icd"),
        valid_hadm_ids,
        admission_lookup,
        args.csv_chunk_rows,
    )
    first_af = first_eligible_af_times(
        admissions,
        af_hadm_ids,
        surgeries,
        args.ohs_washout_days,
    )
    token_to_id, vocabulary = build_vocabulary(
        train_counts, args.min_code_count, args.max_vocab_size
    )
    arrays, normalization = prepare_arrays(
        admissions,
        deaths,
        splits,
        codes_by_hadm,
        domains,
        surgeries,
        first_af,
        token_to_id,
        args.ohs_washout_days,
    )
    split_counts = {
        name: int((arrays["patient_splits"] == split_id).sum())
        for split_id, name in enumerate(("train", "validation", "test"))
    }
    encounter_split_counts = {
        name: int(
            sum(
                arrays["patient_offsets"][index + 1] - arrays["patient_offsets"][index]
                for index in np.flatnonzero(arrays["patient_splits"] == split_id)
            )
        )
        for split_id, name in enumerate(("train", "validation", "test"))
    }
    metadata = {
        "format_version": 1,
        "source": "MIMIC-IV hosp module",
        "cause_names": ["af", "death"],
        "source_directory": str(args.mimic_dir.resolve()),
        "icd_harmonization": (
            "CMS ICD-10-CM to ICD-9-CM GEM; first listed mapping retained; "
            "unmapped ICD-10 codes omitted"
        ),
        "gem_file": str(args.gem.resolve()),
        "gem_archive_member": gem_member,
        "gem_sha256": sha256_file(args.gem),
        "gem_entries": len(gem),
        "diagnosis_mapping_audit": dict(sorted(mapping_audit.items())),
        "patients": int(len(arrays["subject_ids"])),
        "encounters": int(len(arrays["hadm_ids"])),
        "diagnosis_occurrences": int(len(arrays["code_ids"])),
        "vocab_size": int(len(token_to_id)),
        "patient_split_counts": split_counts,
        "encounter_split_counts": encounter_split_counts,
        "risk_eligible_encounters": int(arrays["risk_masks"].sum()),
        "af_events_within_365d": int(
            ((arrays["event_types"] == 1) & arrays["risk_masks"]).sum()
        ),
        "death_events_within_365d": int(
            ((arrays["event_types"] == 2) & arrays["risk_masks"]).sum()
        ),
        "af_outcome_admissions_before_surgery_exclusion": len(af_hadm_ids),
        "af_outcome_patients_after_surgery_exclusion": len(first_af),
        "open_heart_surgery_patients": len(surgeries),
        "ohs_washout_days": args.ohs_washout_days,
        "open_heart_surgery_definition": {
            "procedure_fields_only": True,
            "icd9_cm_procedure_prefixes": list(OHS_ICD9_PREFIXES),
            "icd10_pcs_prefixes": list(OHS_ICD10_PREFIXES),
            "procedure_codes_are_model_features": False,
        },
        "min_code_count": args.min_code_count,
        "split_seed": args.seed,
        "clinical_inputs": [
            "unordered current-encounter diagnosis set",
            "longitudinal encounter timing",
            "standardized age",
            "binary sex",
        ],
        "outcome_codes_removed": [
            "ICD-9 42731/42732",
            "ICD-10 I48* before GEM",
            "mapped ICD-9 42731/42732 after GEM",
        ],
        "data_use": (
            "Derived from credentialed MIMIC-IV data; use is governed by "
            "the PhysioNet data-use agreement."
        ),
        **normalization,
    }
    save_dataset(args.out_dir, arrays, metadata, vocabulary)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
