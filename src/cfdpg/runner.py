"""Pure helpers for the stage runner: command building, output parsing, resume logic.

No LeRobot or LIBERO import here, so everything can be unit-tested anywhere.
The CLI that actually launches training and evaluation is scripts/run_stages.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .results import CellRecord

# Matches the aggregate dict LeRobot prints at the end of lerobot-eval. With a
# single task per call, the last match ("overall") is the cell we want.
_EVAL_RE = re.compile(
    r"'pc_success': ([0-9.]+), 'n_episodes': (\d+), 'n_success': (\d+)"
)


def parse_eval_output(text: str) -> tuple[int, int]:
    """Return (n_rollouts, successes) from the captured lerobot-eval output."""
    matches = _EVAL_RE.findall(text)
    if not matches:
        raise ValueError("no aggregated metrics found in eval output")
    _, n_ep, n_ok = matches[-1]
    return int(n_ep), int(n_ok)


def episodes_flag(episodes: Iterable[int]) -> str:
    eps = sorted(set(int(e) for e in episodes))
    if not eps:
        raise ValueError("empty episode list")
    return "[" + ",".join(map(str, eps)) + "]"


def checkpoint_dir(stage_dir: str | Path, steps: int) -> Path:
    """LeRobot saves checkpoints under checkpoints/<6-digit step>/pretrained_model."""
    return Path(stage_dir) / "checkpoints" / f"{steps:06d}" / "pretrained_model"


def build_train_cmd(
    *,
    episodes: Iterable[int],
    out_dir: str | Path,
    seed: int,
    steps: int,
    batch_size: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    num_integration_steps: int,
    dataset: str = "lerobot/libero",
) -> list[str]:
    return [
        "lerobot-train",
        f"--dataset.repo_id={dataset}",
        f"--dataset.episodes={episodes_flag(episodes)}",
        f"--output_dir={out_dir}",
        f"--seed={seed}",
        "--policy.type=multi_task_dit",
        "--policy.objective=flow_matching",
        "--policy.timestep_sampling_strategy=beta",
        f"--policy.num_integration_steps={num_integration_steps}",
        "--policy.integration_method=euler",
        f"--policy.num_layers={num_layers}",
        f"--policy.hidden_dim={hidden_dim}",
        f"--policy.num_heads={num_heads}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={steps}",
        "--log_freq=100",
        "--wandb.enable=false",
    ]


def build_eval_cmd(
    *,
    policy_path: str | Path,
    suite: str,
    task_id: int,
    out_dir: str | Path,
    seed: int,
    n_episodes: int,
    batch_size: int = 10,
    resolution: int = 256,
) -> list[str]:
    if n_episodes % batch_size != 0:
        raise ValueError("n_episodes must be a multiple of the eval batch size")
    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids=[{task_id}]",
        # Must match the dataset images; the default (360) silently breaks results.
        f"--env.observation_height={resolution}",
        f"--env.observation_width={resolution}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--seed={seed}",
        f"--output_dir={out_dir}",
    ]


def stage_episodes(task_episodes: dict[int, list[int]], order: list[int], stage: int) -> list[int]:
    """Union of the dataset episodes of the first stage+1 tasks of the order."""
    eps: list[int] = []
    for t in order[: stage + 1]:
        eps.extend(task_episodes[t])
    return sorted(set(eps))


def already_done(records: Iterable[CellRecord], run_id: str, stage: int, task: int, kind: str) -> bool:
    return any(
        r.run_id == run_id and r.stage == stage and r.task == task and r.kind == kind
        for r in records
    )
