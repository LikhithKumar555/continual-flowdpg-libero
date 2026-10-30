#!/usr/bin/env python
"""Sanity check for the critic: Q and V along whole episodes (demos or recorded rollouts).

  python q_along_episode.py --policy_path <ckpt>/pretrained_model \
      --root $DATA/spatial_t0_demos --episodes 0 1 2 --out $WORK/q_demo.png

  (without --root it reads lerobot/libero; then --episodes are lerobot/libero episode indices)

With a sparse terminal reward, Q should rise towards the end of a successful episode. The dashed
line is the ideal discounted value gamma^(steps to the success frame). Writes a PNG and a CSV.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flowdpg_patch as fp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--root", default=None, help="local dataset folder (default: lerobot/libero from the Hub)")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="PNG path; the CSV goes next to it")
    ap.add_argument("--fdpg_hidden", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    fp.install(fp.FDPGConfig(hidden=a.fdpg_hidden))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

    policy = MultiTaskDiTPolicy.from_pretrained(a.policy_path).to(a.device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=a.policy_path,
                                      preprocessor_overrides={"device_processor": {"device": a.device}})
    repo_id = f"local/{Path(a.root).name}" if a.root else "lerobot/libero"
    meta = LeRobotDatasetMetadata(repo_id, root=a.root)
    ds = LeRobotDataset(repo_id, root=a.root, episodes=a.episodes,
                        delta_timestamps=resolve_delta_timestamps(policy.config, meta))
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4)

    cfg = policy.config
    n_obs, k = cfg.n_obs_steps, cfg.n_action_steps
    s = n_obs - 1
    rows = []
    with torch.no_grad():
        for batch in loader:
            ep = batch["episode_index"].tolist()
            fr = batch["frame_index"].tolist()
            succ = batch.get("next.success")
            b = policy._prepare_batch(pre(batch))
            cur, _ = fp.split_time(b, n_obs)
            c = policy.observation_encoder.encode(cur)
            a_exec = b[fp.ACTION][:, s:s + k].reshape(c.shape[0], -1)
            q = torch.minimum(*policy.fdpg_q(c, a_exec)).cpu().tolist()
            v = policy.fdpg_v(c).cpu().tolist()
            for i in range(len(ep)):
                sc = bool(succ[i].any()) if succ is not None else None
                rows.append({"episode": ep[i], "frame": fr[i], "q": q[i], "v": v[i], "success_frame": sc})

    gamma = policy._fdpg_hp.gamma
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e in sorted({r["episode"] for r in rows}):
        er = sorted((r for r in rows if r["episode"] == e), key=lambda r: r["frame"])
        f = [r["frame"] for r in er]
        line, = ax.plot(f, [r["q"] for r in er], label=f"episode {e}: Q")
        succ_frames = [r["frame"] for r in er if r["success_frame"]]
        end = succ_frames[0] if succ_frames else (max(f) if er[0]["success_frame"] is None else None)
        if end is not None:
            ax.plot(f, [gamma ** max(0, end - x) for x in f], "--", color=line.get_color(), alpha=0.6,
                    label=f"episode {e}: gamma^(steps to success)")
        else:
            ax.text(f[-1], er[-1]["q"], " failed", color=line.get_color(), fontsize=8)
    ax.set_xlabel("frame")
    ax.set_ylabel("Q(s, executed chunk)")
    ax.set_title(f"Critic along episodes ({Path(a.policy_path).parent.parent.parent.name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out} and {out.with_suffix('.csv')} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
