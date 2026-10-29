#!/usr/bin/env python
"""Map every task of lerobot/libero to its episode indices.

Writes $WORK/task_episodes.json: {task_index: {"text": instruction, "episodes": [...]}}.
Episodes of one task are NOT contiguous in lerobot/libero, so this map is needed to train
on a subset of tasks (--dataset.episodes).
"""
import collections
import json
import os
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = Path(os.environ.get("WORK", ".")) / "task_episodes.json"

ds = LeRobotDataset("lerobot/libero")
meta = ds.meta
ep_idx = np.asarray(ds.hf_dataset["episode_index"])
task_idx = np.asarray(ds.hf_dataset["task_index"])

t = meta.tasks  # DataFrame (index = task text) in newer formats, dict in older ones
if hasattr(t, "iterrows"):
    names = {int(r["task_index"]): str(name) for name, r in t.iterrows()}
else:
    names = {int(k): str(v) for k, v in dict(t).items()}

by_task = collections.defaultdict(set)
for e, k in zip(ep_idx, task_idx):
    by_task[int(k)].add(int(e))

out = {str(k): {"text": names.get(k, "?"), "episodes": sorted(v)} for k, v in sorted(by_task.items())}
OUT.write_text(json.dumps(out, indent=1))
for k, v in sorted(by_task.items()):
    eps = sorted(v)
    print(f"task {k:2d} | {len(eps):3d} eps [{eps[0]}..{eps[-1]}] | {names.get(k, '?')}")
print("distinct tasks:", len(by_task), "->", OUT)
