#!/bin/bash
set -e

LANG=${1:?Usage: ./run_all.sh <language_code> [config.yaml]}
CONFIG=${2:-config.yaml}

echo "=== Preparing data for language: $LANG ==="
python prepare_data.py --language "$LANG" --config "$CONFIG"

echo "=== Training LoRA ==="
python train_lora.py --language "$LANG" --config "$CONFIG"

echo "=== Merging and converting ==="
python merge_and_convert.py --language "$LANG" --config "$CONFIG"

echo "=== Evaluating ==="
python evaluate.py --language "$LANG" --config "$CONFIG" --compare-baseline

echo "=== Done! Results in ./output/$LANG/ ==="
