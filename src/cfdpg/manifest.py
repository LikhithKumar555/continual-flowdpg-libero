"""Dataset manifest: a record of exactly which tasks and episodes were in the
cumulative dataset at each stage. Saved next to every run so results can be
traced back to their data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class TaskEntry:
    task_id: int
    name: str
    n_episodes: int
    n_transitions: int
    source: str = "demos"        # e.g. "demos", "policy_rollouts"
    added_at_stage: int = 0
    episode_ids: list[str] = field(default_factory=list)


class DatasetManifest:
    def __init__(self, order_name: str = "standard"):
        self.order_name = order_name
        self.tasks: list[TaskEntry] = []

    def add_task(self, entry: TaskEntry) -> None:
        if entry.task_id != len(self.tasks):
            raise ValueError(
                f"tasks must be added in order: expected id {len(self.tasks)}, got {entry.task_id}"
            )
        if entry.n_transitions <= 0 or entry.n_episodes <= 0:
            raise ValueError("task must contain at least one episode and one transition")
        entry.added_at_stage = entry.task_id
        self.tasks.append(entry)

    def at_stage(self, stage: int) -> list[TaskEntry]:
        """Tasks visible to training at `stage` (cumulative: tasks 0..stage)."""
        return [t for t in self.tasks if t.task_id <= stage]

    def task_sizes(self, stage: int) -> list[int]:
        return [t.n_transitions for t in self.at_stage(stage)]

    def to_dict(self) -> dict:
        return {"order_name": self.order_name, "tasks": [asdict(t) for t in self.tasks]}

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["fingerprint"] = self.fingerprint()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(order_name=data["order_name"])
        for t in data["tasks"]:
            m.tasks.append(TaskEntry(**t))
        saved = data.get("fingerprint")
        if saved is not None and saved != m.fingerprint():
            raise ValueError("manifest fingerprint mismatch: file was modified after saving")
        return m
