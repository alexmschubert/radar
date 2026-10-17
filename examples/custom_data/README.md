# Adapting RADAR to another dataset

`prepare_data.py` is a small, runnable adapter for encounter-level data. Copy
it and replace the CSV parsing or label construction while keeping its output
arrays unchanged.

The input has one row per encounter and these columns:

| Column | Meaning |
|---|---|
| `subject_id`, `encounter_id` | Unique integer identifiers |
| `split` | `train`, `validation`, or `test`; one split per subject |
| `event_day` | Encounter time on any consistent day scale |
| `age`, `sex` | Age and female 0/male 1 |
| `codes` | Unordered diagnosis tokens separated by `|` |
| `followup_days` | Observed follow-up from this encounter, capped at 365 |
| `event_type` | 0 event-free/censored, 1 cause 1, or 2 cause 2 |
| `risk_eligible` | 1 when the row contributes to the clinical loss |
| `coding_domain` | Optional: -1 ignored, 0 domain A, or 1 domain B |

Prepare the included artificial example and train RADAR:

```bash
python examples/custom_data/prepare_data.py \
  --input examples/custom_data/example_encounters.csv \
  --out-dir prepared/custom-example \
  --cause-1-name outcome \
  --cause-2-name death

python scripts/train.py \
  --data-dir prepared/custom-example \
  --out-dir runs/custom-example

python scripts/predict.py \
  --checkpoint runs/custom-example/checkpoint_best.pt \
  --data-dir prepared/custom-example \
  --out-dir outputs/custom-example
```

The adapter learns its vocabulary and age normalization from training patients
only and stores complete patient trajectories. The two cause names determine
the prediction CSV column names; the diagnosis-set encoder, temporal memory,
and competing-risk calculation remain unchanged.
