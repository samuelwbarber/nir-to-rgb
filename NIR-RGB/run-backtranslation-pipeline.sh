#!/bin/bash
# End-to-end back-translation pipeline.
#
# Phases:
#   0. Wait for Places365-Standard tar to finish downloading (started elsewhere)
#   1. Train reverse RGB->NIR model (~8h)
#   2. Extract Places365 tar + flatten into data/synth-rgb via symlinks (~1h)
#   3. Generate synthetic NIR for every Places365 image (~10h)
#   4. Train forward NIR->RGB with mixed real+synth dataset (~25h)
#   5. Evaluate (writes experiments/eval_all_v2/summary.csv)
#
# Idempotent: each phase writes a guard file; restart skips already-done phases.
# Designed to be launched with nohup so it survives ssh disconnect.

set -uo pipefail
cd /vol/bitbucket/sb1522/vm-backup/NIR-RGB
PIPE=experiments/backtranslation_pipeline
mkdir -p "$PIPE"
LOG="$PIPE/pipeline.log"
PY=./.venv-kaist/bin/python

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "================================================================="
log "PIPELINE STARTED on $(hostname)"
log "================================================================="
log "GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

# ---------------------------------------------------------------------
# Phase 0: wait for Places365 download
# ---------------------------------------------------------------------
TAR=/vol/bitbucket/sb1522/places365/train_256_places365standard.tar
EXPECT_TAR_SIZE=26103685120
phase=0
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] waiting for $TAR to finish downloading (~24 GB)"
  while true; do
    if [[ -f "$TAR" ]]; then
      sz=$(stat -c '%s' "$TAR" 2>/dev/null || echo 0)
      if (( sz >= EXPECT_TAR_SIZE )); then
        log "[phase $phase] download complete: $sz bytes"
        break
      fi
      log "[phase $phase] still downloading: $sz / $EXPECT_TAR_SIZE bytes ($(( sz * 100 / EXPECT_TAR_SIZE ))%)"
    else
      log "[phase $phase] tar not yet present"
    fi
    sleep 120
  done
  touch "$PIPE/phase${phase}.done"
fi

# ---------------------------------------------------------------------
# Phase 1: train reverse RGB->NIR (~8 hours)
# ---------------------------------------------------------------------
phase=1
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] training reverse RGB->NIR model"
  if ! $PY -u scripts/train.py --config configs/rgb_to_nir_v1.yaml 2>&1 | tee -a "$LOG"; then
    log "[phase $phase] FAILED — exiting (re-run pipeline to retry from this phase)"
    exit 1
  fi
  if [[ ! -f experiments/rgb_to_nir_v1/checkpoints/best.pth ]]; then
    log "[phase $phase] FAILED — best.pth missing after training"
    exit 1
  fi
  touch "$PIPE/phase${phase}.done"
  log "[phase $phase] done"
fi

# ---------------------------------------------------------------------
# Phase 2: extract Places365 + flat symlinks (~1 hour)
# ---------------------------------------------------------------------
phase=2
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] extracting Places365 + flattening into data/synth-rgb"
  if ! $PY -u scripts/setup_places365.py 2>&1 | tee -a "$LOG"; then
    log "[phase $phase] FAILED — exiting"
    exit 1
  fi
  touch "$PIPE/phase${phase}.done"
  log "[phase $phase] done"
fi

# ---------------------------------------------------------------------
# Phase 3: generate synthetic NIR for every Places365 image (~10 hours)
# ---------------------------------------------------------------------
phase=3
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] generating synthetic NIRs (~1.8M images)"
  if ! $PY -u scripts/generate_synthetic_nir.py 2>&1 | tee -a "$LOG"; then
    log "[phase $phase] FAILED — exiting"
    exit 1
  fi
  touch "$PIPE/phase${phase}.done"
  log "[phase $phase] done"
fi

# ---------------------------------------------------------------------
# Phase 4: train forward NIR->RGB on real + synthetic (~25 hours)
# ---------------------------------------------------------------------
phase=4
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] training forward NIR->RGB on real+synthetic"
  if ! $PY -u scripts/train.py --config configs/ablation_09_backtranslation.yaml 2>&1 | tee -a "$LOG"; then
    log "[phase $phase] FAILED — exiting"
    exit 1
  fi
  if [[ ! -f experiments/ablation_09_backtranslation/checkpoints/best.pth ]]; then
    log "[phase $phase] FAILED — best.pth missing after training"
    exit 1
  fi
  touch "$PIPE/phase${phase}.done"
  log "[phase $phase] done"
fi

# ---------------------------------------------------------------------
# Phase 5: final evaluation (~10 min)
# ---------------------------------------------------------------------
phase=5
if [[ ! -f "$PIPE/phase${phase}.done" ]]; then
  log "[phase $phase] running eval_all.py (writes experiments/eval_all_v2/)"
  if ! $PY -u eval_all.py 2>&1 | tee -a "$LOG"; then
    log "[phase $phase] FAILED — but training artifacts are preserved"
    exit 1
  fi
  touch "$PIPE/phase${phase}.done"
  log "[phase $phase] done"
fi

log "================================================================="
log "PIPELINE COMPLETE"
log "Results: experiments/eval_all_v2/summary.csv"
log "================================================================="
