"""CLARE-style lower-triangular heatmap (rows = stages, columns = tasks)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_triangle(
    r: np.ndarray,
    path: str | Path,
    title: str = "",
    task_labels: list[str] | None = None,
    annotate: bool = True,
) -> Path:
    r = np.asarray(r, dtype=float)
    m = r.shape[0]
    labels = task_labels or [f"T{i + 1}" for i in range(m)]

    fig, ax = plt.subplots(figsize=(0.7 * m + 2.5, 0.7 * m + 2))
    masked = np.ma.masked_invalid(r)
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")

    ax.set_xticks(range(m), labels, rotation=45, ha="right")
    ax.set_yticks(range(m), [f"after {l}" for l in labels])
    ax.set_xlabel("evaluated task")
    ax.set_ylabel("training stage")
    if title:
        ax.set_title(title)

    if annotate:
        for i in range(m):
            for j in range(m):
                if not np.isnan(r[i, j]):
                    colour = "white" if r[i, j] < 0.6 else "black"
                    ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=8, color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("success rate")
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
