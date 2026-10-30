#!/usr/bin/env python
"""Create the starting dataset for a task (or several): its demos from lerobot/libero, in the
shared schema with reward / success / done labels (1 / True on the last frame of each demo).

  python build_task_dataset.py --suite libero_spatial --task_ids 0 --out $DATA/spatial_t0_demos

The result is a local LeRobot dataset. Train on it with
  --dataset.repo_id=local/spatial_t0_demos --dataset.root=$DATA/spatial_t0_demos
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_data as ld  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_ids", type=int, nargs="+", required=True, help="env task ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_id", default=None, help="default: local/<out folder name>")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = []
    for t in a.task_ids:
        text, e = ld.episodes_for_env_task(a.suite, t)
        print(f"{a.suite} task {t}: {len(e)} demos | {text}", file=sys.stderr)
        eps += e
    out = Path(a.out)
    repo_id = a.repo_id or f"local/{out.name}"
    ds = ld.build_demo_dataset(LeRobotDataset(ld.SRC_REPO), eps, out, repo_id)
    print(f"wrote {ds.meta.total_episodes} episodes / {ds.meta.total_frames} frames -> {out}")
    print("features:", sorted(ds.meta.features))
    print(f"train with: --dataset.repo_id={repo_id} --dataset.root={out}")


if __name__ == "__main__":
    main()
