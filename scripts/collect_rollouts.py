#!/usr/bin/env python
"""Run a policy in LIBERO and save EVERY rollout (successes and failures) as a LeRobot dataset.

  python collect_rollouts.py --policy_path <ckpt>/pretrained_model --suite libero_spatial --task_id 0 \
      --n_episodes 50 --batch_size 10 --seed 1000 --out $DATA/spatial_t0_rollouts_r1

Recorded per frame, in the same schema as the demos (libero_data.py):
  the observation exactly as the policy sees it after LeRobot's LIBERO processing (flipped
  cameras, 8-d state) - i.e. the demo dataset's format, at 256x256 -, the action in dataset
  units (after the policy postprocessor), next.reward, next.success, next.done, task.
Success flags and episode ends come from LeRobot's own rollout() (same logic as lerobot-eval);
reward = 1 on the success step, 0 otherwise.
Also writes <out>/episodes.jsonl (seed, success, length per episode) and prints the success rate.
Seeds: episode j uses --seed + j; use a fresh seed range per round so rollouts differ from evals.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_data as ld  # noqa: E402


class Capture:
    """Wrap a processor pipeline: return its output unchanged, hand a copy to `sink`."""

    def __init__(self, inner, sink):
        self._inner = inner
        self._sink = sink

    def __call__(self, x):
        y = self._inner(x)
        self._sink(y)
        return y

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_id", default=None)
    ap.add_argument("--flowdpg", action=argparse.BooleanOptionalAction, default=True,
                    help="checkpoint contains the FlowDPG critic (use --no-flowdpg for plain BC checkpoints)")
    ap.add_argument("--fdpg_hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.n_episodes % a.batch_size:
        raise SystemExit("--n_episodes must be a multiple of --batch_size")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new folder per round")

    if a.flowdpg:
        from flowdpg_patch import FDPGConfig, install
        install(FDPGConfig(hidden=a.fdpg_hidden))

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    src_meta = LeRobotDatasetMetadata(ld.SRC_REPO)
    features = ld.rollout_features(src_meta.features)
    cam_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    h, w = features[cam_keys[0]]["shape"][:2] if features[cam_keys[0]]["shape"][-1] == 3 \
        else features[cam_keys[0]]["shape"][1:]

    env_cfg = LiberoEnv(task=a.suite, task_ids=[a.task_id], observation_height=int(h), observation_width=int(w))
    envs = make_env(env_cfg, n_envs=a.batch_size, use_async_envs=False)
    vec_env = next(iter(next(iter(envs.values())).values()))

    policy_cfg = PreTrainedConfig.from_pretrained(a.policy_path)
    policy_cfg.pretrained_path = a.policy_path
    policy_cfg.device = a.device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg, pretrained_path=a.policy_path,
        preprocessor_overrides={"device_processor": {"device": a.device}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    obs_steps: list[dict] = []
    act_steps: list = []

    def keep_obs(o):
        missing = [k for k in cam_keys + [ld.OBS_STATE] if k not in o]
        if missing:
            raise RuntimeError(f"processed observation lacks {missing}; has {sorted(o)}")
        obs_steps.append({k: o[k].detach().cpu() if hasattr(o[k], "detach") else o[k]
                          for k in cam_keys + [ld.OBS_STATE]})

    def keep_action(act):
        act_steps.append(act.detach().cpu() if hasattr(act, "detach") else act)

    ds = LeRobotDataset.create(repo_id=a.repo_id or f"local/{out.name}", fps=src_meta.fps, features=features,
                               root=out, robot_type=src_meta.robot_type,
                               use_videos=any(v["dtype"] == "video" for v in features.values()))
    task = list(vec_env.call("task_description"))[0]
    summary = []
    n_batches = a.n_episodes // a.batch_size
    try:
        for b in range(n_batches):
            seeds = [a.seed + b * a.batch_size + i for i in range(a.batch_size)]
            obs_steps.clear()
            act_steps.clear()
            with torch.no_grad():
                ret = rollout(env=vec_env, policy=policy, env_preprocessor=Capture(env_pre, keep_obs),
                              env_postprocessor=env_post, preprocessor=pre,
                              postprocessor=Capture(post, keep_action), seeds=seeds)
            n_steps = ret["done"].shape[1]
            if len(obs_steps) != n_steps or len(act_steps) != n_steps:
                raise RuntimeError(f"captured {len(obs_steps)} obs / {len(act_steps)} actions for {n_steps} steps")
            successes = ret["success"].numpy()
            for i in range(a.batch_size):
                d = ld.first_done(ret["done"][i].numpy())
                for frame in ld.episode_frames(obs_steps, act_steps, i, successes, d, task, cam_keys):
                    ds.add_frame(frame)
                ds.save_episode()
                ok = bool(np.asarray(successes[i][: d + 1]).any())
                summary.append({"episode_index": len(summary), "seed": seeds[i], "success": ok, "length": d + 1})
            done_eps = summary[-a.batch_size:]
            print(f"batch {b + 1}/{n_batches}: {sum(e['success'] for e in done_eps)}/{a.batch_size} successes",
                  flush=True)
    finally:
        ds.finalize()
        with open(out / "episodes.jsonl", "w") as f:
            for e in summary:
                f.write(json.dumps(e) + "\n")

    n_ok = sum(e["success"] for e in summary)
    print(f"collected {len(summary)} episodes, {n_ok} successes ({100 * n_ok / max(1, len(summary)):.0f}%), "
          f"{sum(e['length'] for e in summary)} frames -> {out}")


if __name__ == "__main__":
    main()
