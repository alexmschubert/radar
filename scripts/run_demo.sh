#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(dirname "$script_dir")"
demo_root="${1:-demo_run}"

python "$repository_dir/scripts/make_demo.py" --out-dir "$demo_root/mimic"
python "$repository_dir/scripts/prepare_mimic.py" \
  --mimic-dir "$demo_root/mimic" \
  --gem "$demo_root/mimic/resources/icd10_to_icd9_gem.csv" \
  --out-dir "$demo_root/prepared" \
  --min-code-count 1 \
  --seed 0
python "$repository_dir/scripts/train.py" \
  --data-dir "$demo_root/prepared" \
  --out-dir "$demo_root/model" \
  --epochs 1 \
  --batch-patients 4 \
  --embedding-dim 16 \
  --memory-channels 4 \
  --attention-heads 4 \
  --domain-hidden-dim 8 \
  --dropout 0 \
  --half-lives-days 7 30 365 36500 \
  --device cpu
python "$repository_dir/scripts/predict.py" \
  --checkpoint "$demo_root/model/checkpoint_best.pt" \
  --data-dir "$demo_root/prepared" \
  --out-dir "$demo_root/predictions" \
  --device cpu

echo "Demo complete: $demo_root/predictions/predictions.csv"
