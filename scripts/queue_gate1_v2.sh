#!/usr/bin/env bash
# Gate 1 re-run after the supervisor's review: split-projection critic + uniform flow time.
# LIBERO-Spatial env task 0, BC vs FlowDPG, seeds 0-2, each = train + 50-episode eval.
# BC is re-run with uniform t too, so both methods differ only in the critic push.
#   STEPS=10000 bash queue_gate1_v2.sh      (set STEPS from the long-run checkpoint result)
# Finished runs/cells are skipped, so the queue can be restarted after an interruption.
set -u
STEPS=${STEPS:-10000}
S=$(cd "$(dirname "$0")" && pwd)/run_stages.py
LOG=/tmp/${USER}_cfdpg_logs/queue_gate1_v2.log
mkdir -p "$(dirname "$LOG")"

run() {
  local what=$1; shift
  echo "$(date '+%F %T')  START $what" | tee -a "$LOG"
  if python "$S" --suite libero_spatial --order 0 --timestep uniform --steps "$STEPS" "$@" >> "$LOG" 2>&1; then
    echo "$(date '+%F %T')  DONE  $what" | tee -a "$LOG"
  else
    echo "$(date '+%F %T')  FAIL  $what (see $LOG)" | tee -a "$LOG"
  fi
}

for SEED in 0 1 2; do
  run "flowdpg split, seed $SEED" --algo flowdpg --seed "$SEED"
  run "bc, seed $SEED"            --algo bc      --seed "$SEED"
done
echo "$(date '+%F %T')  QUEUE FINISHED" | tee -a "$LOG"
cat "$WORK"/runs/*_tu*_libero_spatial_o0_s*/cells.jsonl
