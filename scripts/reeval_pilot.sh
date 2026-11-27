#!/usr/bin/env bash
# After the pilot queue: re-evaluate every triangle cell of the five pilot runs with 50 episodes
# (about +-7% per cell instead of +-15%). Waits until queue_pilot.sh has finished, so it never
# shares the GPU with a training. Restartable.
#   tmux new -s reeval   then   bash scripts/reeval_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f queue_pilot.sh > /dev/null; do
  echo "$(date '+%a %H:%M') queue still running, checking again in 10 min"; sleep 600
done
RUNS=()
for m in fdpg_cumulative fixed fdpg_cold fdpg_current bc_cumulative; do
  RUNS+=("$DATA/continual/${m}_libero_spatial_o0-1-2_s0")
done
python scripts/reeval.py --episodes 50 "${RUNS[@]}"
for r in "${RUNS[@]}"; do
  echo "--- $(basename "$r")"
  python scripts/report.py "$r/cells_e50.jsonl"
done
