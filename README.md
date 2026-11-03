# Continual FlowDPG on LIBERO

Offline FlowDPG on LeRobot's MultiTaskDiT flow-matching policy, evaluated on LIBERO, plus the
tooling for the cumulative-data continual experiment (stage runner, CLARE-style triangle metrics).

Status (29 Oct 2026): single-task flow-matching BC and single-task offline FlowDPG run end to end.
Rollout collection (successes and failures, with reward / success labels), the demo dataset builder,
dataset merging and the critic Q-value diagnostic are implemented; the offline-to-online loop on one
task is the current milestone.

## Results so far

All runs: MultiTaskDiT (4 layers, hidden 512, 8 heads), flow matching with beta timestep sampling,
10 Euler steps at inference, batch 32, 10,000 steps, seed 0, 50 evaluation episodes at
256x256, one RTX 4090 (fafnir).

| Suite, task | Method | Success | 95% interval (Wilson) |
| --- | --- | --- | --- |
| LIBERO-Spatial, env task 0 | FlowDPG (offline, from scratch) | 86% (43/50) | 74-93% |
| LIBERO-Spatial, env task 0 | Flow-matching BC | 74% (37/50) | 60-84% |
| LIBERO-Object, env task 0 | Flow-matching BC | 82% (41/50) | 69-90% |

One seed per method: the Spatial difference is not significant (Fisher exact p = 0.21).
Seeds 1-2 and a no-push control are in `scripts/queue_gate1.sh`.

**Evaluation pitfall:** LeRobot's LIBERO env defaults to 360x360 images, the `lerobot/libero`
dataset is 256x256. MultiTaskDiT center-crops 224 at inference, so at 360 it sees a different
field of view. The same Object checkpoint scored 0% at 360 and 82% at 256. Always pass
`--env.observation_height=256 --env.observation_width=256`.

## FlowDPG implementation (`scripts/flowdpg_patch.py`)

Applied as a runtime patch to `MultiTaskDiTPolicy`, so LeRobot's trainer, data loading,
normalisation, checkpointing and `lerobot-eval` are reused unchanged
(`scripts/flowdpg_launch.py train|eval ...`).

- **Transitions at chunk level.** The policy predicts a chunk of H = 32 actions and executes
  K = 24 (`n_action_steps`). The patch extends `observation_delta_indices` by K, so each sample
  also carries the observation window at t+K. The critic scores the K executed actions.
- **Reward / termination.** Sparse success reward (1 on the success step, 0 otherwise).
  If the dataset has `next.reward` (rollouts, and demos prepared with `build_task_dataset.py`) the
  patch loads its K-step window: r = sum_j gamma^j r_{t+j}, terminal = any success in the window
  (LIBERO ends an episode on success). Episodes that end without success (time limit) are not
  terminal: V(s') is bootstrapped from the last frame. Without a reward column (raw `lerobot/libero`,
  all demos successful) the demo end inside the chunk is taken as the success
  (`observation.state_is_pad` at t+K). gamma = 0.99 per step, chunk discount gamma^K.
- **Critic (IQL).** Twin Q(s, a_exec) and V(s) on the policy's own conditioning vector **detached**
  (critic does not train the CLIP/state encoders). The conditioning vector is large
  ((8 state + 2x768 CLIP + 512 text) x 2 steps = 4112) compared with the action chunk (24 x 7 = 168).
  Default `critic_arch=split`: obs -> 256 and action -> 256 through separate projections, then a
  2x512 MLP, so the action is not drowned out. `critic_arch=concat` (one MLP on [obs, action], used
  for all runs before 3 Nov) is kept; scripts read the architecture from the checkpoint.
  V: expectile 0.7 on min target-Q of the data action. Q: MSE to r + gamma^K (1 - done) V(s').
  Target Q by Polyak averaging (tau = 0.005) in `update()`. Critic lr 3e-4 (own param group).
- **Actor (FlowDPG).** With LeRobot's path x_t = t a + (1 - (1 - sigma) t) x0 and
  u_t = a - (1 - sigma) x0:
  - a_hat = x_t + (1 - t) v(x_t, t, s)
  - g = d min(Q1, Q2)(s, a_hat_exec) / d a_hat, zero outside the executed slice
  - a* = a_hat + alpha * ||u_t|| / (||g|| + eps) * g,  u* = a* - (1 - sigma) x0
  - L = lam ||v - u*||^2 + (1 - lam) ||v - u_t||^2 + w_c ||a_hat - a||^2
  - alpha = 0.5, lam ramps 0 -> 0.5 over 2,000 steps, w_c = 1.0. No backprop through the ODE.
- `--fdpg.mode=bc_critic` keeps critic and consistency term but sets lam = 0 (control run).
- Flow-time sampling: LeRobot's default is a beta distribution that favours small t (noisy
  inputs); the FlowDPG paper samples t uniformly. `run_stages.py --timestep uniform` sets it.
- Logged per step: `fdpg_bc, fdpg_rl, fdpg_cons, fdpg_lambda, fdpg_v_loss, fdpg_q_loss,
  fdpg_q_mean, fdpg_v_mean, fdpg_done_frac, fdpg_grad_norm`.

Open points for review: the exact FlowDPG target (u* = a* - x0) and the consistency term follow
my reading of the paper; there is no reference implementation to compare against.

## Layout

```
scripts/
  flowdpg_patch.py         FlowDPG + IQL critic as a patch on MultiTaskDiT
  flowdpg_launch.py        lerobot-train / lerobot-eval with the patch (--fdpg.* flags)
  run_stages.py            cumulative retrain-from-scratch stage runner (bc | flowdpg | bc_critic),
                           restartable, one JSON line per triangle cell
  queue_gate1.sh           overnight queue: seeds and control on Spatial task 0
  list_tasks.py            task -> episode map for lerobot/libero (writes task_episodes.json)
  episodes_for_env_task.py episode list for one env task, matched by instruction text
  report.py                triangle heatmap + AUC / FWT / NBT from a cells.jsonl
  libero_data.py           shared dataset schema (cameras, 8-d state, action, next.reward/success/done)
  build_task_dataset.py    demos of chosen tasks -> local dataset with reward/success/done labels
  collect_rollouts.py      run a policy in LIBERO, save every episode (incl. failures) in that schema
  merge_data.py            merge demos + rollout rounds into one training dataset
  q_along_episode.py       critic diagnostic: Q and V along demos / rollouts vs gamma^(steps to success)
src/cfdpg/                 results store, metrics, heatmap, samplers, dataset manifest
tests/                     59 tests (metrics, samplers, runner, FlowDPG patch on a stand-in, rollout frames)
notes/experiment_log.md
```

## Setup (fafnir)

```bash
export WORK=/techfak/user/$USER/work          # code; large data/checkpoints -> /media/internal
git clone https://github.com/huggingface/lerobot.git $WORK/lerobot
cd $WORK/lerobot && python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[libero]" -e ".[multi_task_dit]" -e ".[training]"
export MUJOCO_GL=egl
python /path/to/this/repo/scripts/list_tasks.py     # once; writes $WORK/task_episodes.json
```

LeRobot commit used: `<fill in: git -C $WORK/lerobot rev-parse HEAD>`
(PyTorch 2.11.0+cu130, Python 3.12.3, driver 580, CUDA 13.0).

## Running

BC on one task (train + 50-episode eval, result in `runs/<run_id>/cells.jsonl`):
```bash
python scripts/run_stages.py --algo bc --suite libero_spatial --order 0 --seed 0
```
FlowDPG on the same task:
```bash
python scripts/run_stages.py --algo flowdpg --suite libero_spatial --order 0 --seed 0
# extra settings: --fdpg alpha=0.3 lam_max=0.5
```
Cumulative stages (task 0, then 0+1, then 0+1+2, each from scratch):
```bash
python scripts/run_stages.py --algo flowdpg --suite libero_object --order 0 1 2 --seed 0
python scripts/report.py $WORK/runs/<run_id>/cells.jsonl
```
FlowDPG checkpoints contain the critic, so evaluate them through the launcher:
`python scripts/flowdpg_launch.py eval --policy.path=... <lerobot-eval args>`.

Offline-to-online data flow for one task (storage on fafnir's HDD):
```bash
export DATA=/media/internal/hdd/$USER/data
python scripts/build_task_dataset.py --suite libero_spatial --task_ids 0 --out $DATA/spatial_t0_demos
python scripts/collect_rollouts.py --policy_path <ckpt>/pretrained_model --suite libero_spatial --task_id 0 \
    --n_episodes 50 --batch_size 10 --seed 1000 --out $DATA/spatial_t0_rollouts_r1
python scripts/merge_data.py --inputs $DATA/spatial_t0_demos $DATA/spatial_t0_rollouts_r1 --out $DATA/spatial_t0_r1
python scripts/q_along_episode.py --policy_path <ckpt>/pretrained_model --root $DATA/spatial_t0_rollouts_r1 \
    --episodes 0 1 2 --out $WORK/q_rollouts_r1.png
```
Rollouts record the observation exactly as the policy sees it after LeRobot's LIBERO processing
(the demo dataset's format, 256x256), so demos and rollouts merge into one training set.

Tests: `pip install numpy matplotlib pytest torch && python -m pytest`. The dataset pipeline
(build -> record -> merge -> load with the patch's reward window) was also checked end to end with
LeRobot's dataset code on synthetic data; collection inside the LIBERO simulator is untested outside fafnir.
