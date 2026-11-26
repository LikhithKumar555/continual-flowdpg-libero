#!/usr/bin/env bash
# Pilot baselines on LIBERO-Spatial tasks 0 1 2, seed 0, one after another (one run fills the 4090).
# fixed and fdpg_cold reuse stage 0 of the finished main run; fdpg_current and bc_cumulative train
# their own (current warm-starts from stage 0's optimizer state, which the main run already cleaned up).
# Every run pauses around the reserved GPU hours (continual.py --avoid, default Mon/Thu/Fri 14-17).
# Restartable: run it again after an interruption and finished parts are skipped.
#   tmux new -s queue   then   bash scripts/queue_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ORDER="0 1 2"
MAIN="$DATA/continual/fdpg_cumulative_libero_spatial_o0-1-2_s0"
[ -f "$MAIN/stages.jsonl" ] || { echo "main run not found: $MAIN"; exit 1; }

run() {   # method, extra args...
  local m=$1; shift
  echo "=== $(date '+%a %d %b %H:%M') start $m"
  python scripts/continual.py --method "$m" --suite libero_spatial --order $ORDER --seed 0 "$@" \
    2>&1 | tee -a "$DATA/pilot_$m.log"
  echo "=== $(date '+%a %d %b %H:%M') done $m"
}

run fixed --stage0_from "$MAIN"
run fdpg_cold --stage0_from "$MAIN"
run fdpg_current
run bc_cumulative

for m in fdpg_cumulative fixed fdpg_cold fdpg_current bc_cumulative; do
  echo "--- $m"
  python scripts/report.py "$DATA/continual/${m}_libero_spatial_o0-1-2_s0/cells.jsonl"
done
