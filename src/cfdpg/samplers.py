"""Minibatch sampling over a growing multi-task dataset.

Two rules, selectable by config:
  uniform  : every transition equally likely -> big tasks dominate
  balanced : pick a task uniformly, then a transition inside it -> every task
             gets equal attention, small tasks are replayed more often

The sampler returns (task_index, local_index) pairs. The caller maps them to
real data. RNG state can be saved and restored so a killed run resumes with
the same sample stream.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MODES = ("uniform", "balanced")


class CumulativeSampler:
    def __init__(self, task_sizes: Sequence[int], mode: str = "uniform", seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        sizes = np.asarray(task_sizes, dtype=np.int64)
        if sizes.ndim != 1 or len(sizes) == 0:
            raise ValueError("task_sizes must be a non-empty 1-D sequence")
        if (sizes <= 0).any():
            raise ValueError("every task needs at least one transition")
        self.sizes = sizes
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    @property
    def n_tasks(self) -> int:
        return len(self.sizes)

    def task_probabilities(self) -> np.ndarray:
        """Probability that a sampled transition belongs to each task."""
        if self.mode == "uniform":
            return self.sizes / self.sizes.sum()
        return np.full(self.n_tasks, 1.0 / self.n_tasks)

    def add_task(self, size: int) -> None:
        """Grow the dataset by one task (called at the start of each stage)."""
        if size <= 0:
            raise ValueError("new task needs at least one transition")
        self.sizes = np.append(self.sizes, size)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.mode == "uniform":
            flat = self.rng.integers(0, self.sizes.sum(), size=batch_size)
            offsets = np.concatenate(([0], np.cumsum(self.sizes)))
            tasks = np.searchsorted(offsets, flat, side="right") - 1
            local = flat - offsets[tasks]
        else:
            tasks = self.rng.integers(0, self.n_tasks, size=batch_size)
            local = (self.rng.random(batch_size) * self.sizes[tasks]).astype(np.int64)
        return tasks, local

    def get_state(self) -> dict:
        return {
            "mode": self.mode,
            "sizes": self.sizes.tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def set_state(self, state: dict) -> None:
        self.mode = state["mode"]
        self.sizes = np.asarray(state["sizes"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng"]
