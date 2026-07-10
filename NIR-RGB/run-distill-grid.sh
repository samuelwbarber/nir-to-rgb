#!/bin/bash
set -uo pipefail
cd /vol/bitbucket/sb1522/vm-backup/NIR-RGB
OUT=experiments/distill_comparison_grid.png
LOG=experiments/distill_comparison_grid.log

echo "=== STARTED $(date) ===" | tee "$LOG"
echo "host: $(hostname)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

./.venv-kaist/bin/python -u scripts/render_distill_comparison_grid.py \
  --out "$OUT" \
  2>&1 | tee -a "$LOG"

ec=${PIPESTATUS[0]}
echo "=== EXITED $(date) (exit $ec) ===" | tee -a "$LOG"
exit $ec
