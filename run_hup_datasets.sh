#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-352}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT_DIR/pretrained_pth/pvt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/model_pth_hup}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

DATASETS=(
  kvasir
  lung
  isic_2018
  etis_polyp
  covid19
  colondb_polyp
  clinicdb_polyp
)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "ERROR: pretrained directory not found: $PRETRAINED_DIR" >&2
  echo "Set PRETRAINED_DIR=/path/to/pvt before running this script." >&2
  exit 1
fi

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$ROOT_DIR/data/$dataset"
  for split in train val; do
    [[ -d "$dataset_dir/$split/images" ]] || {
      echo "ERROR: missing $dataset_dir/$split/images" >&2
      exit 1
    }
    [[ -d "$dataset_dir/$split/masks" ]] || {
      echo "ERROR: missing $dataset_dir/$split/masks" >&2
      exit 1
    }
  done
done

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$ROOT_DIR/data/$dataset"
  run_dir="$OUTPUT_DIR/$dataset"
  mkdir -p "$run_dir"

  echo
  echo "========== HUPAnno: $dataset on CUDA device $GPU_ID =========="
  "$PYTHON_BIN" -u train_polyp_hup.py \
    --epoch "$EPOCHS" \
    --batchsize "$BATCH_SIZE" \
    --test_batchsize "$TEST_BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --pretrained_dir "$PRETRAINED_DIR" \
    --train_image_root "$dataset_dir/train/images" \
    --train_mask_root "$dataset_dir/train/masks" \
    --val_image_root "$dataset_dir/val/images" \
    --val_mask_root "$dataset_dir/val/masks" \
    --train_save "$run_dir"

  metrics_file="$run_dir/best_metrics.json"
  [[ -f "$metrics_file" ]] || {
    echo "ERROR: expected metrics file was not created: $metrics_file" >&2
    exit 1
  }

  "$PYTHON_BIN" - "$dataset" "$metrics_file" <<'PY'
import json
import sys

dataset, metrics_file = sys.argv[1:]
with open(metrics_file) as handle:
    metrics = json.load(handle)

print(f"BEST {dataset}")
print(f"  checkpoint: {metrics['checkpoint']}")
print(f"  epoch:      {metrics['best_epoch']}")
print(f"  samples:    {metrics['n_samples']}")
print(f"  Mean Dice:   {metrics['mean_dice']:.6f}")
print(f"  Median Dice: {metrics['median_dice']:.6f}")
print(f"  Std Dice:    {metrics['std_dice']:.6f}")
print(f"  Mean IoU:    {metrics['mean_iou']:.6f}")
print(f"  Median IoU:  {metrics['median_iou']:.6f}")
print(f"  Mean HD95:   {metrics['mean_hd95']:.6f}")
print(f"  Median HD95: {metrics['median_hd95']:.6f}")
PY
done

echo
echo "All requested datasets completed. Checkpoint and metrics are under: $OUTPUT_DIR"
