"""Raw per-cell evaluation records and the lower-triangular result matrix.

Rule: raw cells are the source of truth. Metrics and plots are always derived
from these records, never stored on their own, and files are append-only.

Indexing convention (0-based):
  stage m  = after finishing training on task m
  task  n  = the task being evaluated
  kind "post"  : evaluation after training stage m, for tasks n <= m
  kind "entry" : evaluation of task m *before* its first update (zero-shot
                 transfer from the previous stage); stored with stage == task
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

KINDS = ("post", "entry")


@dataclass(frozen=True)
class CellRecord:
    run_id: str
    method: str
    seed: int
    order: str          # name of the task order, e.g. "standard" or "perm1"
    stage: int
    task: int
    kind: str           # "post" or "entry"
    n_rollouts: int
    successes: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive")
        if not 0 <= self.successes <= self.n_rollouts:
            raise ValueError("successes must be within [0, n_rollouts]")
        if self.stage < 0 or self.task < 0:
            raise ValueError("stage and task must be non-negative")
        if self.kind == "post" and self.task > self.stage:
            raise ValueError("post-evaluation only covers tasks seen so far (task <= stage)")
        if self.kind == "entry" and self.task != self.stage:
            raise ValueError("entry evaluation must have task == stage")

    @property
    def rate(self) -> float:
        return self.successes / self.n_rollouts


class ResultStore:
    """Append-only JSONL store. One line per evaluated cell."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CellRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def read(self) -> list[CellRecord]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(CellRecord(**json.loads(line)))
        return out


def build_matrix(
    records: Iterable[CellRecord],
    n_tasks: int,
    kind: str = "post",
) -> np.ndarray:
    """Return an (n_tasks x n_tasks) matrix of success rates.

    Row = stage, column = task. Cells not evaluated stay NaN, so the upper
    triangle (tasks not yet seen) is NaN by construction. Duplicate cells for
    the same (stage, task) are an error, which catches double-logging bugs.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    mat = np.full((n_tasks, n_tasks), np.nan)
    for r in records:
        if r.kind != kind:
            continue
        if r.stage >= n_tasks or r.task >= n_tasks:
            raise ValueError(f"record outside {n_tasks}x{n_tasks} matrix: {r}")
        if not np.isnan(mat[r.stage, r.task]):
            raise ValueError(f"duplicate cell for stage={r.stage}, task={r.task}")
        mat[r.stage, r.task] = r.rate
    return mat
