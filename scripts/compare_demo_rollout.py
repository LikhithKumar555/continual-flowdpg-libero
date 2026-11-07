#!/usr/bin/env python
"""Side-by-side check that rollout frames look like demo frames (same camera, orientation,
state convention, action convention, task text). Training on successful rollouts alone
collapsed the policy, so a format mismatch that min/max stats cannot show is suspected.

  python compare_demo_rollout.py --demos $DATA/spatial_t0_demos \
      --rollouts $DATA/online_spatial_t0_s0_v4/rollouts_r1 --out $WORK/compare.png

Prints: task strings, start-of-episode state (demo mean vs rollout mean, per dimension),
action statistics incl. how binary the gripper command is, and episode lengths.
Saves a PNG: for 3 demo and 3 successful-rollout episodes, both cameras at the first frame and
at the middle frame, so orientation / camera assignment / colour can be compared by eye.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f"local/{Path(root).name}", root=root)


def episodes(ds):
    ep = np.asarray(ds.hf_dataset["episode_index"])
    out = {}
    for i, e in enumerate(ep):
        out.setdefault(int(e), []).append(i)
    return out


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dem, rol = load(a.demos), load(a.rollouts)
    cams = [k for k, v in dem.meta.features.items() if v.get("dtype") in ("video", "image")]
    ok = {}
    ep_file = Path(a.rollouts) / "episodes.jsonl"
    if ep_file.exists():
        ok = {json.loads(l)["episode_index"]: json.loads(l)["success"] for l in open(ep_file)}
    d_eps, r_eps = episodes(dem), episodes(rol)
    r_succ = [e for e in r_eps if ok.get(e, True)]
    print(f"demos: {len(d_eps)} episodes | rollouts: {len(r_eps)} episodes, {len(r_succ)} successful")

    print("\nTASK TEXT")
    print("  demo   :", repr(dem[d_eps[min(d_eps)][0]]["task"]))
    print("  rollout:", repr(rol[r_eps[min(r_eps)][0]]["task"]))

    print("\nEPISODE LENGTH (frames)")
    dl = [len(v) for v in d_eps.values()]
    rl = [len(r_eps[e]) for e in r_succ]
    print(f"  demos   mean {np.mean(dl):6.1f}  min {min(dl)}  max {max(dl)}")
    print(f"  rollouts (successful) mean {np.mean(rl):6.1f}  min {min(rl)}  max {max(rl)}")

    def start_states(ds, eps):
        return np.stack([to_np(ds[idx[0]]["observation.state"]) for idx in eps])

    ds0 = start_states(dem, d_eps.values())
    rs0 = start_states(rol, [r_eps[e] for e in r_succ])
    print("\nSTATE AT THE FIRST FRAME (mean ± std over episodes)  dims: 0-2 pos, 3-5 rotation, 6-7 fingers")
    for i in range(ds0.shape[1]):
        print(f"  dim {i}: demos {ds0[:, i].mean():8.3f} ± {ds0[:, i].std():.3f}   "
              f"rollouts {rs0[:, i].mean():8.3f} ± {rs0[:, i].std():.3f}")

    def actions(ds, eps, limit=40):
        idx = [i for e in list(eps)[:limit] for i in e]
        return np.stack([to_np(ds[i]["action"]) for i in idx])

    da = actions(dem, d_eps.values())
    ra = actions(rol, [r_eps[e] for e in r_succ])
    print("\nACTIONS (mean ± std over frames)  dims: 0-5 arm, 6 gripper")
    for i in range(da.shape[1]):
        print(f"  dim {i}: demos {da[:, i].mean():7.3f} ± {da[:, i].std():.3f}   "
              f"rollouts {ra[:, i].mean():7.3f} ± {ra[:, i].std():.3f}")
    for name, arr in (("demos", da), ("rollouts", ra)):
        g = arr[:, -1]
        print(f"  gripper {name}: {100 * np.mean(np.isclose(np.abs(g), 1.0, atol=0.02)):.0f}% of frames at ±1, "
              f"{100 * np.mean(g > 0):.0f}% positive")

    rows = [("demo", dem, list(d_eps.values())[:3]), ("rollout", rol, [r_eps[e] for e in r_succ[:3]])]
    fig, axes = plt.subplots(6, 2 * len(cams), figsize=(3 * 2 * len(cams), 3 * 6))
    r = 0
    for name, ds, eps in rows:
        for idx in eps:
            for j, frame in enumerate((idx[0], idx[len(idx) // 2])):
                item = ds[frame]
                for c, cam in enumerate(cams):
                    img = to_np(item[cam])
                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        img = np.transpose(img, (1, 2, 0))
                    ax = axes[r, j * len(cams) + c]
                    ax.imshow(np.clip(img, 0, 1))
                    ax.set_title(f"{name} ep{int(item['episode_index'])} "
                                 f"{'start' if j == 0 else 'middle'} {cam.split('.')[-1]}", fontsize=8)
                    ax.axis("off")
            r += 1
    fig.tight_layout()
    fig.savefig(a.out, dpi=80)
    print(f"\nsaved image grid: {a.out}  (rows 1-3 demos, rows 4-6 successful rollouts)")


if __name__ == "__main__":
    main()
