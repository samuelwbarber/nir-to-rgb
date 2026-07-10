#!/bin/bash
set -uo pipefail
cd /vol/bitbucket/sb1522/vm-backup/NIR-RGB
EXP=student_l1b_a_baseline
LOG=/vol/bitbucket/sb1522/vm-backup/NIR-RGB/experiments/${EXP}/train.log
mkdir -p "$(dirname "$LOG")"

echo "=== STARTED $(date) ===" | tee -a "$LOG"
echo "host: $(hostname)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

./.venv-kaist/bin/python -u scripts/train.py \
  --config configs/${EXP}.yaml \
  2>&1 | tee -a "$LOG"

ec=${PIPESTATUS[0]}
echo "=== EXITED $(date) (exit $ec) ===" | tee -a "$LOG"
exit $ec
