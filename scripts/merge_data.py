#!/usr/bin/env python
"""Merge local datasets (demos + rollout rounds) into one training dataset.

  python merge_data.py --inputs $DATA/spatial_t0_demos $DATA/rollouts_r1 --out $DATA/spatial_t0_r1

All inputs must share the schema of libero_data.py (build_task_dataset.py / collect_rollouts.py).
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_id", default=None)
    a = ap.parse_args()

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --out")
    srcs = [LeRobotDataset(repo_id=f"local/{Path(p).name}", root=p) for p in a.inputs]
    for p, d in zip(a.inputs, srcs):
        print(f"{p}: {d.meta.total_episodes} episodes, {d.meta.total_frames} frames")
    merged = merge_datasets(srcs, output_repo_id=a.repo_id or f"local/{out.name}", output_dir=out,
                            concatenate_videos=False)
    print(f"merged: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames -> {out}")


if __name__ == "__main__":
    main()
