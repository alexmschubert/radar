# Prepared data schema

RADAR stores patient trajectories as aligned NumPy arrays in one directory.
The arrays are uncompressed so they can be memory-mapped during training.

| File | Level | Meaning |
|---|---|---|
| `code_ids.npy` | diagnosis occurrence | Integer diagnosis vocabulary IDs |
| `code_offsets.npy` | encounter + 1 | CSR offsets into `code_ids` |
| `patient_offsets.npy` | patient + 1 | CSR offsets into encounters |
| `event_days.npy` | encounter | Encounter time in integer-like days |
| `ages.npy` | encounter | Training-standardized age |
| `sexes.npy` | encounter | Female 0, male 1 |
| `diagnosis_counts.npy` | encounter | Number of raw diagnosis rows (detached domain-classifier input) |
| `coding_domains.npy` | encounter | Auxiliary domain label 0/1, or -1 to ignore |
| `followup_days.npy` | encounter | Exact observed exposure, capped at 365 |
| `event_types.npy` | encounter | Event-free/censored 0, first cause 1, second cause 2 |
| `risk_masks.npy` | encounter | Whether the row contributes clinical loss |
| `labels_365d.npy` | encounter | Binary first-cause outcome within 365 days |
| `label_masks_365d.npy` | encounter | Whether the binary 365-day outcome is observable for AUROC |
| `subject_ids.npy` | patient | Source patient identifiers |
| `hadm_ids.npy` | encounter | Source encounter identifiers (MIMIC `hadm_id`) |
| `patient_splits.npy` | patient | Train 0, validation 1, test 2 |

`metadata.json` records preparation choices and normalization statistics.
`vocabulary.json` contains the training-derived ICD-9-CM token vocabulary.
ICD-10-CM diagnoses are harmonized with the supplied CMS GEM before vocabulary
construction; unmapped codes and AF/flutter outcome codes are omitted.

Patient histories must be complete, contiguous, and chronological. A patient
must never be divided across dataset splits or training batches.

Binary evaluation masks are separate from survival-training masks: shorter
censored follow-up contributes exposure to the likelihood but is omitted from
AUROC. The MIMIC adapter marks binary outcomes observable when AF occurs within
365 days or a subsequent encounter establishes at least 365 days of follow-up,
subject to the incident-risk and surgery exclusions. The MIMIC preparation
command writes both binary evaluation arrays.

The two binary evaluation arrays are optional for custom datasets. If absent,
the loader derives labels from `event_types == 1` and evaluates eligible rows
with a first-cause event or `followup_days >= 365`; shorter follow-up, including
early competing events, is omitted. Validation must contain both classes.
