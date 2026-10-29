#!/usr/bin/env bash
# Gate 1 queue on LIBERO-Spatial task 0: runs one after another, each = train 10k steps + 50-episode eval.
# Already finished runs/cells are skipped, so the queue can simply be restarted after an interruption.
# Usage (inside tmux, venv active):  bash $WORK/scripts/queue_gate1.sh
set -u
S=$WORK/scripts/run_stages.py
LOG=/tmp/${USER}_cfdpg_logs/queue_gate1.log
mkdir -p "$(dirname "$LOG")"

run() {  # $1 = description, rest = run_stages arguments
  local what=$1; shift
  echo "$(date '+%F %T')  START $what" | tee -a "$LOG"
  if python "$S" --suite libero_spatial --order 0 "$@" >> "$LOG" 2>&1; then
    echo "$(date '+%F %T')  DONE  $what" | tee -a "$LOG"
  else
    echo "$(date '+%F %T')  FAIL  $what (see $LOG)" | tee -a "$LOG"
  fi
}

run "flowdpg seed 1"           --algo flowdpg   --seed 1
run "bc seed 1"                --algo bc        --seed 1
run "bc_critic (no push) s0"   --algo bc_critic --seed 0
run "flowdpg seed 2"           --algo flowdpg   --seed 2
run "bc seed 2"                --algo bc        --seed 2

echo "$(date '+%F %T')  QUEUE FINISHED" | tee -a "$LOG"
cat "$WORK"/runs/*_libero_spatial_o0_s*/cells.jsonl
