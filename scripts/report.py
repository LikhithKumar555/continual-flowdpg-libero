#!/usr/bin/env python
"""Build the triangle heatmap and metrics from a results JSONL file."""

import argparse
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfdpg.metrics import summarize  # noqa: E402
from cfdpg.plotting import plot_triangle  # noqa: E402
from cfdpg.results import ResultStore, build_matrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
a = ap.parse_args()

recs = ResultStore(a.jsonl).read()
n = max(r.stage for r in recs) + 1
post = build_matrix(recs, n, "post")
entry = build_matrix(recs, n, "entry") if any(r.kind == "entry" for r in recs) else None
# Task 0 has no previous stage, so its entry cell does not exist; use tasks 1..N-1.
entry_vec = None
if entry is not None and n >= 2:
    entry_vec = np.diag(entry)[1:]
summary = summarize(post, entry_vec)
print(json.dumps(summary, indent=2))
png = Path(a.jsonl).with_suffix(".png")
plot_triangle(post, png, title=Path(a.jsonl).stem)
print("saved", png)
