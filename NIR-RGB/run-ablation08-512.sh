#!/bin/bash
set -uo pipefail
cd /vol/bitbucket/sb1522/vm-backup/NIR-RGB
LOG=experiments/ablation_08_512_from512base/train.log
mkdir -p experiments/ablation_08_512_from512base
PY=./.venv-kaist/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== STARTED $(date) on $(hostname) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

if ! $PY -u scripts/train.py --config configs/ablation_08_512_from512base.yaml 2>&1 | tee -a "$LOG"; then
  echo "=== TRAINING FAILED $(date) ===" | tee -a "$LOG"
  exit 1
fi

echo "=== DONE $(date) ===" | tee -a "$LOG"
