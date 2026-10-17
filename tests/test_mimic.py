from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from radar.mimic import (
    continuous_competing_target,
    in_washout,
    is_ohs,
    load_gem,
    prepare_arrays,
    read_diagnoses,
    read_surgeries,
)


def test_loads_official_cms_gem_zip_and_uses_first_mapping(tmp_path):
    archive_path = tmp_path / "2018_I10gem.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "2018_I10gem.txt",
            "A000 0010 00000\nI10 4019 00000\nI10 4010 10011\nZ999 NoDx 01000\n",
        )

    mapping, member = load_gem(archive_path)

    assert member == "2018_I10gem.txt"
    assert mapping["A000"] == "0010"
    assert mapping["I10"] == "4019"
    assert mapping["Z999"] == "NODX"


def test_translation_excludes_native_and_mapped_af_codes(tmp_path):
    diagnoses = tmp_path / "diagnoses.csv"
    pd.DataFrame(
        [
            {"hadm_id": 1, "icd_code": "401.9", "icd_version": 9},
            {"hadm_id": 1, "icd_code": "I10", "icd_version": 10},
            {"hadm_id": 1, "icd_code": "I48.91", "icd_version": 10},
            {"hadm_id": 1, "icd_code": "Z00.0", "icd_version": 10},
            {"hadm_id": 1, "icd_code": "X00", "icd_version": 10},
        ]
    ).to_csv(diagnoses, index=False)

    codes, af_hadm_ids, domains, counts, audit = read_diagnoses(
        diagnoses,
        {1},
        {1: 0},
        {"I10": "4019", "I4891": "42731", "Z000": "42732"},
        2,
    )

    assert codes[1] == {"ICD9_4019"}
    assert af_hadm_ids == {1}
    assert domains[1] == (1, 4)
    assert counts == {"ICD9_4019": 1}
    assert audit["outcome_codes_removed_before_mapping"] == 1
    assert audit["outcome_codes_removed_after_mapping"] == 1
    assert audit["icd10_rows_unmapped_and_omitted"] == 1


def test_open_heart_surgery_lists_cover_icd9_and_icd10_pcs():
    assert is_ohs("3521", 9)
    assert is_ohs("02RF0JZ", 10)
    assert not is_ohs("3893", 9)
    assert not is_ohs("0W3P0ZZ", 10)


def test_open_heart_surgery_washout_boundaries():
    surgery = (pd.Timestamp("2020-01-01"),)
    assert in_washout(pd.Timestamp("2020-01-01"), surgery, 90)
    assert in_washout(pd.Timestamp("2020-03-31"), surgery, 90)
    assert not in_washout(pd.Timestamp("2020-04-01"), surgery, 90)
    assert not in_washout(pd.Timestamp("2020-01-01"), surgery, 0)


def test_leading_zeros_survive_single_row_csv_chunks(tmp_path):
    diagnosis_path = tmp_path / "diagnoses.csv"
    diagnosis_path.write_text("hadm_id,icd_code,icd_version\n1,0010,9\n")
    codes, _, _, _, _ = read_diagnoses(diagnosis_path, {1}, {1: 0}, {}, 1)
    assert codes[1] == {"ICD9_0010"}

    procedure_path = tmp_path / "procedures.csv"
    procedure_path.write_text(
        "hadm_id,chartdate,icd_code,icd_version\n1,2020-01-01,02RF0JZ,10\n"
    )
    surgeries = read_surgeries(
        procedure_path, {1}, {1: (10, pd.Timestamp("2020-01-01"))}, 1
    )
    assert surgeries == {10: (pd.Timestamp("2020-01-01"),)}


def test_af_within_one_day_is_outside_incident_risk_set():
    admissions = pd.DataFrame(
        {
            "subject_id": [10, 10],
            "hadm_id": [1, 2],
            "admittime": pd.to_datetime(
                ["2020-01-01", "2020-01-01 12:00"], format="mixed"
            ),
            "age": [50.0, 50.0],
            "sex_male": [False, False],
        }
    )
    arrays, _ = prepare_arrays(
        admissions,
        {},
        {10: 0},
        {},
        {},
        {},
        {10: pd.Timestamp("2020-01-01 12:00")},
        {"<UNK>": 0},
        90,
    )
    assert not arrays["risk_masks"].any()
    assert not arrays["followup_days"].any()


@pytest.mark.parametrize(
    "last,af,death,eligible,expected",
    [
        (60, 120, None, True, (120.0, 1, True)),
        (60, None, 120, True, (120.0, 2, True)),
        (60, 120, 120, True, (120.0, 1, True)),
        (60, 180, 120, True, (120.0, 2, True)),
        (60, 365, None, True, (365.0, 1, True)),
        (60, None, 500, True, (365.0, 0, True)),
        (60, None, None, True, (60.0, 0, True)),
        (0, None, None, True, (0.0, 0, True)),
        (60, 120, None, False, (0.0, 0, False)),
    ],
)
def test_exact_exposure_targets(last, af, death, eligible, expected):
    assert (
        continuous_competing_target(
            index_day=0,
            last_encounter_day=last,
            af_day=af,
            death_day=death,
            risk_eligible=eligible,
        )
        == expected
    )


@pytest.mark.parametrize(
    "af_delay,death_delay,expected_labels,expected_masks",
    [
        (None, None, [False, False], [False, False]),
        (None, 500, [False, False], [False, False]),
        (100, None, [True, False], [True, False]),
        (365, None, [True, True], [True, True]),
        (500, None, [False, False], [False, False]),
    ],
)
def test_binary_evaluation_uses_observable_outcomes(
    af_delay, death_delay, expected_labels, expected_masks
):
    start = pd.Timestamp("2020-01-01")
    admissions = pd.DataFrame(
        {
            "subject_id": [10, 10],
            "hadm_id": [1, 2],
            "admittime": [start, start + pd.to_timedelta(100, unit="D")],
            "age": [50.0, 50.3],
            "sex_male": [False, False],
        }
    )
    deaths = (
        {}
        if death_delay is None
        else {10: start + pd.to_timedelta(death_delay, unit="D")}
    )
    first_af = (
        {} if af_delay is None else {10: start + pd.to_timedelta(af_delay, unit="D")}
    )
    arrays, _ = prepare_arrays(
        admissions, deaths, {10: 0}, {}, {}, {}, first_af, {"<UNK>": 0}, 90
    )
    assert arrays["labels_365d"].tolist() == expected_labels
    assert arrays["label_masks_365d"].tolist() == expected_masks
    if death_delay == 500:
        # Survival exposure uses known event dates; binary evaluation uses
        # observed AF diagnoses or subsequent encounters.
        assert arrays["followup_days"].tolist() == [365.0, 365.0]
        assert arrays["risk_masks"].all()
