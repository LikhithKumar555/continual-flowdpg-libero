#!/usr/bin/env python
"""Check that a merged dataset contains exactly the frames of its sources, in order.

  python check_merged.py --merged $DATA/online_spatial_t0_s0_nopush3/data_r1 \
      --sources $DATA/spatial_t0_demos $DATA/online_spatial_t0_s0_nopush3/rollouts_r1

Every policy trained on a merged dataset (demos + rollouts) collapsed, while training on the demo
dataset alone did not, so this compares, frame by frame, image(s), state, action and episode
labels of the merged dataset with the source datasets (merged frame offset + i == source frame i).
"""
import argparse
from pathlib import Path

import numpy as np


def load(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f"local/{Path(root).name}", root=root)


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--per_source", type=int, default=40, help="frames checked per source")
    a = ap.parse_args()

    m = load(a.merged)
    cams = [k for k, v in m.meta.features.items() if v.get("dtype") in ("video", "image")]
    keys = cams + ["observation.state", "action", "episode_success", "next.reward"]
    offset = 0
    rng = np.random.default_rng(0)
    print(f"merged: {len(m)} frames, {m.meta.total_episodes} episodes")
    total_bad = 0
    for src_root in a.sources:
        s = load(src_root)
        n = len(s)
        idx = sorted(set(rng.choice(n, size=min(a.per_source, n), replace=False).tolist()) | {0, n - 1})
        worst = {k: 0.0 for k in keys}
        bad = 0
        for i in idx:
            si, mi = s[i], m[offset + i]
            frame_bad = False
            for k in keys:
                if k not in si or k not in mi:
                    continue
                d = float(np.abs(to_np(si[k]).astype(np.float32) - to_np(mi[k]).astype(np.float32)).max())
                worst[k] = max(worst[k], d)
                # images: decoding the same video twice may differ by a few gray levels at most
                if d > (0.05 if k in cams else 1e-5):
                    frame_bad = True
            bad += frame_bad
        total_bad += bad
        print(f"{Path(src_root).name}: {n} frames at merged offset {offset}; checked {len(idx)}, "
              f"{bad} mismatching")
        print("   largest difference per key:", {k: round(v, 4) for k, v in worst.items()})
        offset += n
    if offset != len(m):
        print(f"WARNING: sources have {offset} frames in total, merged has {len(m)}")
    print("RESULT:", "merged dataset matches its sources" if total_bad == 0 and offset == len(m)
          else "MISMATCH: the merged dataset does not reproduce its sources")


if __name__ == "__main__":
    main()
