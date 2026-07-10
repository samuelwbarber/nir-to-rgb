#!/bin/bash
# Watchdog: relaunch ablation_15a/b on siegfried after reboot / process death.
# Invoked every 5 min by a systemd --user timer on shell1.
set -uo pipefail

SIEGFRIED=siegfried.doc.ic.ac.uk
REPO=/vol/bitbucket/sb1522/vm-backup/NIR-RGB
LOG_DIR=$REPO/experiments/_watchdog
LOG=$LOG_DIR/siegfried_watchdog.log
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" >> "$LOG"; }

SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10)

if ! ssh "${SSH_OPTS[@]}" "$SIEGFRIED" "true" 2>/dev/null; then
  log "siegfried unreachable; skipping (likely rebooting)"
  exit 0
fi

ssh "${SSH_OPTS[@]}" "$SIEGFRIED" \
  "loginctl show-user sb1522 2>/dev/null | grep -q Linger=yes || loginctl enable-linger sb1522 >/dev/null 2>&1" \
  2>/dev/null

# Counts only the actual python training process, not the bash/pgrep that ran
# this very command. We match comm=python under user sb1522 and then check each
# PID's /proc/<pid>/cmdline for the slug. This avoids the self-match bug where
# pgrep -f would see its own argv (which embeds the search pattern verbatim).
count_alive() {
  local slug=$1
  ssh "${SSH_OPTS[@]}" "$SIEGFRIED" \
    "n=0; for pid in \$(pgrep -u sb1522 -x python 2>/dev/null); do \
       tr '\\0' ' ' < /proc/\$pid/cmdline 2>/dev/null | grep -q '${slug}' && n=\$((n+1)); \
     done; echo \$n" 2>/dev/null || echo 0
}

check_and_relaunch() {
  local slug=$1
  local script=$2
  local cuda_dev=$3
  local count
  count=$(count_alive "$slug")
  if [ "${count:-0}" -gt 0 ]; then
    return 0
  fi
  log "ablation_${slug}: DEAD - relaunching (CUDA_VISIBLE_DEVICES=$cuda_dev via $script)"
  # `-n -f` makes ssh redirect stdin from /dev/null and go to background after
  # auth, so the local ssh returns immediately. `nohup setsid ...` on the remote
  # detaches the bash from the ssh session. The earlier `& disown` pattern hung
  # because non-interactive bash retained the ssh channel's stdio.
  ssh -n -f "${SSH_OPTS[@]}" "$SIEGFRIED" \
    "cd $REPO && CUDA_VISIBLE_DEVICES=$cuda_dev nohup setsid bash $script </dev/null >/dev/null 2>&1" \
    2>/dev/null
  sleep 20
  count=$(count_alive "$slug")
  if [ "${count:-0}" -gt 0 ]; then
    log "ablation_${slug}: relaunch OK"
  else
    log "ablation_${slug}: relaunch FAILED - manual intervention needed"
  fi
}

check_and_relaunch 15a_l1boost_split42_s1 run-ablation15a.sh 0
check_and_relaunch 15b_l1boost_split42_s2 run-ablation15b.sh 1
