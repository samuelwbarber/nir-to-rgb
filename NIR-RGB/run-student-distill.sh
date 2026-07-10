#!/bin/bash
set -uo pipefail
cd /vol/bitbucket/sb1522/vm-backup/NIR-RGB
LOG=experiments/student_08_distill_downstream/train.log
mkdir -p experiments/student_08_distill_downstream
PY=./.venv-kaist/bin/python

echo "=== STARTED $(date) on $(hostname) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

if ! $PY -u scripts/train.py --config configs/student_08_distill_downstream.yaml 2>&1 | tee -a "$LOG"; then
  echo "=== TRAINING FAILED $(date) ===" | tee -a "$LOG"
  exit 1
fi

echo "=== TRAINING DONE — exporting ONNX ===" | tee -a "$LOG"
$PY -u scripts/export_onnx.py --config configs/student_08_distill_downstream.yaml 2>&1 | tee -a "$LOG"

echo "=== DONE $(date) ===" | tee -a "$LOG"
