#!/usr/bin/env bash
# Evaluate every saved checkpoint of one training run on the SAME 50 starting positions (eval seed 0),
# e.g. to see whether training longer than 10k steps helps.
#   bash eval_checkpoints.sh <run_output_dir> <suite> <task_id> [flowdpg]
#   bash eval_checkpoints.sh $WORK/runs/bc_spatial_t0_long_s0 libero_spatial 0
# Add "flowdpg" as 4th argument for checkpoints that contain the FlowDPG critic.
set -u
RUN=$1; SUITE=$2; TASK=$3; KIND=${4:-bc}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=/tmp/${USER}_cfdpg_logs; mkdir -p "$LOGDIR"
OUT=$RUN/ckpt_eval.jsonl
for CK in $(ls -d "$RUN"/checkpoints/[0-9]*/ | sort); do
  STEP=$(basename "$CK")
  if [ -f "$OUT" ] && grep -q "\"step\": \"$STEP\"" "$OUT"; then echo "skip $STEP (done)"; continue; fi
  if [ "$KIND" = "flowdpg" ]; then EVAL=(python "$HERE/flowdpg_launch.py" eval); else EVAL=(lerobot-eval); fi
  LOG=$LOGDIR/ckpt_eval_$(basename "$RUN")_$STEP.log
  "${EVAL[@]}" --policy.path="$CK/pretrained_model" --env.type=libero --env.task="$SUITE" \
    --env.task_ids="[$TASK]" --env.observation_height=256 --env.observation_width=256 \
    --eval.batch_size=10 --eval.n_episodes=50 --seed=0 --output_dir="$RUN/ckpt_eval/$STEP" > "$LOG" 2>&1
  RES=$(grep -o "'n_episodes': [0-9]*, 'n_success': [0-9]*" "$LOG" | tail -1)
  N=$(echo "$RES" | grep -o "n_episodes': [0-9]*" | grep -o "[0-9]*$")
  K=$(echo "$RES" | grep -o "n_success': [0-9]*" | grep -o "[0-9]*$")
  if [ -z "$N" ]; then echo "step $STEP: eval FAILED, see $LOG"; continue; fi
  echo "{\"step\": \"$STEP\", \"n_rollouts\": $N, \"successes\": $K}" >> "$OUT"
  echo "step $STEP: $K/$N successes ($((100 * K / N))%)"
done
echo "all results: $OUT"
