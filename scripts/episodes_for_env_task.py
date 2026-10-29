#!/usr/bin/env python
"""Print the lerobot/libero episode list for one LIBERO env task, matched by instruction text.

  python episodes_for_env_task.py libero_spatial 0      -> [1272,1273,...]

The env's task_ids and the dataset's task_index are in different orders, so tasks are
matched by their language instruction. Info goes to stderr; the list is the last stdout line
(LIBERO prints an "[info]" line on import, so use `| tail -n 1` when capturing it).
"""
import json
import os
import sys
from pathlib import Path

from libero.libero import benchmark

TASK_EPISODES = Path(os.environ.get("WORK", ".")) / "task_episodes.json"

suite, task_id = sys.argv[1], int(sys.argv[2])
text = benchmark.get_benchmark_dict()[suite]().get_task(task_id).language
d = json.load(open(TASK_EPISODES))
hits = [(k, v) for k, v in d.items() if v["text"].strip().lower() == text.strip().lower()]
print("env task text :", text, file=sys.stderr)
if len(hits) != 1:
    print("NO UNIQUE MATCH, matches:", [k for k, _ in hits], file=sys.stderr)
    sys.exit(1)
k, v = hits[0]
print("dataset task  :", k, "|", len(v["episodes"]), "episodes", file=sys.stderr)
print("[" + ",".join(map(str, v["episodes"])) + "]")
