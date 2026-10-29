"""Continual-learning metrics on a lower-triangular success matrix.

r[m, n] = success of task n after training stage m (defined for n <= m).

Definitions used here (state these explicitly in the report):
  FWT (diagonal)  mean_n r[n, n]               CLARE/LIBERO-style "learned right after training"
  AUC             mean_n  mean_{m>=n} r[m, n]  average success from arrival to the end
  NBT             mean_{n<M-1} mean_{m>n} (ref_n - r[m, n])
                  ref_n = r[n, n] ("diag", LIBERO convention) or the best value
                  seen up to stage m-1 ("peak"). Lower is better, 0 = no forgetting.
  entry success   success on task n BEFORE its first update (zero-shot), kept
                  separate from FWT because the papers use "FWT" for different things.
"""

from __future__ import annotations

import numpy as np


def _check(r: np.ndarray) -> int:
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("matrix must be square (stages x tasks)")
    m = r.shape[0]
    for stage in range(m):
        for task in range(m):
            seen = task <= stage
            if seen and np.isnan(r[stage, task]):
                raise ValueError(f"missing cell: stage={stage}, task={task}")
            if not seen and not np.isnan(r[stage, task]):
                raise ValueError(f"unexpected value above diagonal: stage={stage}, task={task}")
    return m


def forward_transfer_diag(r: np.ndarray) -> float:
    m = _check(r)
    return float(np.mean([r[n, n] for n in range(m)]))


def auc(r: np.ndarray) -> float:
    m = _check(r)
    return float(np.mean([np.mean(r[n:, n]) for n in range(m)]))


def nbt(r: np.ndarray, reference: str = "diag") -> float:
    """Negative backward transfer (forgetting). Needs at least two stages."""
    m = _check(r)
    if m < 2:
        raise ValueError("NBT needs at least two stages")
    if reference not in ("diag", "peak"):
        raise ValueError("reference must be 'diag' or 'peak'")
    per_task = []
    for n in range(m - 1):
        drops = []
        for stage in range(n + 1, m):
            ref = r[n, n] if reference == "diag" else np.max(r[n:stage, n])
            drops.append(ref - r[stage, n])
        per_task.append(np.mean(drops))
    return float(np.mean(per_task))


def entry_success(entry: np.ndarray) -> float:
    """Mean zero-shot success. `entry` is a length-M vector (or the diagonal of an entry matrix)."""
    e = np.asarray(entry, dtype=float)
    if e.ndim == 2:
        e = np.diag(e)
    if np.isnan(e).any():
        raise ValueError("entry vector has missing values")
    return float(np.mean(e))


def final_average(r: np.ndarray) -> float:
    m = _check(r)
    return float(np.mean(r[m - 1, :]))


def worst_task(r: np.ndarray) -> float:
    m = _check(r)
    return float(np.min(r[m - 1, :]))


def summarize(r: np.ndarray, entry: np.ndarray | None = None) -> dict:
    out = {
        "fwt_diag": forward_transfer_diag(r),
        "auc": auc(r),
        "final_avg": final_average(r),
        "worst_task": worst_task(r),
    }
    if np.asarray(r).shape[0] >= 2:
        out["nbt_diag"] = nbt(r, "diag")
        out["nbt_peak"] = nbt(r, "peak")
    if entry is not None:
        out["entry_success"] = entry_success(entry)
    return out


def aggregate_over_seeds(summaries: list[dict]) -> dict:
    """Mean and standard error per metric. Always also report per-seed values."""
    if not summaries:
        raise ValueError("no summaries to aggregate")
    keys = summaries[0].keys()
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in summaries], dtype=float)
        n = len(vals)
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[k] = {"mean": float(np.mean(vals)), "se": se, "per_seed": vals.tolist()}
    return out
