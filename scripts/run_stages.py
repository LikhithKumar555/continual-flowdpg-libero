#!/usr/bin/env python
"""Cumulative retrain-from-scratch stage runner (BC baseline, flow-matching MultiTaskDiT).

Stage m: train a fresh policy on the demos of tasks order[0..m] (uniform over frames),
then evaluate every task order[0..m] ("post" cells). Before training stage m>=1 the
previous checkpoint is evaluated on the new task order[m] ("entry" cell).

Every cell is appended as one JSON line to <out>/<run_id>/cells.jsonl (fields match the
scaffold's CellRecord). Finished trainings and evaluated cells are skipped on re-run,
so the script can be restarted after a disconnect.

Example (on fafnir, inside tmux, venv active):
  python $WORK/scripts/run_stages.py --suite libero_spatial --order 0 1 2 --seed 0
Dry run (prints commands only):  add --dry_run
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/techfak/user/lshivakumar/work"))
TASK_EPISODES = WORK / "task_episodes.json"  # written by list_tasks.py
LAUNCHER = Path(__file__).resolve().parent / "flowdpg_launch.py"

RES = ["--env.observation_height=256", "--env.observation_width=256"]  # MUST match training data
POLICY_FLAGS = [
    "--policy.type=multi_task_dit",
    "--policy.objective=flow_matching",
    "--policy.timestep_sampling_strategy=beta",
    "--policy.num_integration_steps=10",
    "--policy.integration_method=euler",
    "--policy.num_layers=4",
    "--policy.hidden_dim=512",
    "--policy.num_heads=8",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
]

RESULT_RE = re.compile(r"'n_episodes': (\d+), 'n_success': (\d+)")


def env_task_text(suite: str, task_id: int) -> str:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().get_task(task_id).language


def dataset_episodes(suite: str, task_id: int) -> list[int]:
    text = env_task_text(suite, task_id).strip().lower()
    d = json.load(open(TASK_EPISODES))
    hits = [v for v in d.values() if v["text"].strip().lower() == text]
    if len(hits) != 1:
        raise SystemExit(f"no unique dataset task for env task {task_id} ({text!r}): {len(hits)} matches")
    return hits[0]["episodes"]


def parse_overall(log_text: str):
    """Last (n_episodes, n_success) pair in an eval log = the 'overall' block."""
    m = RESULT_RE.findall(log_text)
    if not m:
        return None
    n, k = m[-1]
    return int(n), int(k)


LOCAL_LOGS = Path(f"/tmp/{os.environ.get('USER', 'user')}_cfdpg_logs")


def run(cmd, log_path: Path, dry: bool, run_id: str = ""):
    """Run a command, logging to local disk (network volume can raise BlockingIOError),
    then copy the log next to the run. Progress bars are disabled (TQDM_DISABLE)."""
    print(">>", " ".join(cmd), flush=True)
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    local = LOCAL_LOGS / (f"{run_id}__{log_path.name}" if run_id else log_path.name)
    print(f"   live log: {local}", flush=True)
    env = {**os.environ, "MUJOCO_GL": "egl", "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    with open(local, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    shutil.copy(local, log_path)
    return p.returncode


def load_done(cells_path: Path):
    done = {}
    if cells_path.exists():
        for line in open(cells_path):
            r = json.loads(line)
            done[(r["stage"], r["task"], r["kind"])] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--order", type=int, nargs="+", required=True, help="env task ids in arrival order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algo", default="bc", choices=["bc", "flowdpg", "bc_critic"],
                    help="bc = plain lerobot; flowdpg / bc_critic go through flowdpg_launch.py")
    ap.add_argument("--fdpg", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra FlowDPG settings, e.g. --fdpg alpha=0.3 lam_max=0.5")
    ap.add_argument("--method", default=None, help="name used in run ids (default: <algo>_cumulative_scratch)")
    ap.add_argument("--steps", type=int, default=10000, help="training steps per stage (fixed budget)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=10)
    ap.add_argument("--out", default=str(WORK / "runs"))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    a.method = a.method or f"{a.algo}_cumulative_scratch"
    if a.algo == "bc":
        if a.fdpg:
            raise SystemExit("--fdpg settings need --algo flowdpg or bc_critic")
        train_prefix, eval_prefix = ["lerobot-train"], ["lerobot-eval"]
    else:
        fdpg_flags = [f"--fdpg.mode={a.algo}"] + [f"--fdpg.{kv}" for kv in a.fdpg]
        hidden = [f for f in fdpg_flags if f.startswith("--fdpg.hidden=")]
        train_prefix = [sys.executable, str(LAUNCHER), "train", *fdpg_flags]
        eval_prefix = [sys.executable, str(LAUNCHER), "eval", *hidden]

    run_id = f"{a.method}_{a.suite}_o{''.join(map(str, a.order))}_s{a.seed}"
    root = Path(a.out) / run_id
    cells_path = root / "cells.jsonl"
    done = load_done(cells_path)
    n_tasks = len(a.order)

    def record(stage, task_pos, kind, n, k):
        rec = dict(run_id=run_id, method=a.method, seed=a.seed, order=a.order, stage=stage,
                   task=task_pos, kind=kind, n_rollouts=n, successes=k)
        done[(stage, task_pos, kind)] = rec
        if not a.dry_run:
            root.mkdir(parents=True, exist_ok=True)
            with open(cells_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def evaluate(ckpt: Path, task_pos: int, tag: str):
        out = root / "eval" / tag
        cmd = [*eval_prefix, f"--policy.path={ckpt}", "--env.type=libero", f"--env.task={a.suite}",
               f"--env.task_ids=[{a.order[task_pos]}]", *RES,
               f"--eval.batch_size={a.eval_batch}", f"--eval.n_episodes={a.eval_episodes}",
               f"--seed={a.seed}", f"--output_dir={out}"]
        log = root / "logs" / f"eval_{tag}.log"
        rc = run(cmd, log, a.dry_run, run_id)
        if a.dry_run:
            return 0, 0
        res = parse_overall(open(log).read()) if rc == 0 else None
        if res is None:
            raise SystemExit(f"eval failed or unparsable: see {log}")
        return res

    for m in range(n_tasks):
        prev_ckpt = root / f"stage{m-1}" / "checkpoints" / f"{a.steps:06d}" / "pretrained_model"
        # entry cell: previous model on the new task, before it is trained on
        if m >= 1 and (m, m, "entry") not in done:
            n, k = evaluate(prev_ckpt, m, f"entry_s{m}_t{m}")
            record(m, m, "entry", n, k)

        # train from scratch on the cumulative data
        eps = sorted({e for t in a.order[: m + 1] for e in dataset_episodes(a.suite, t)})
        ckpt = root / f"stage{m}" / "checkpoints" / f"{a.steps:06d}" / "pretrained_model"
        stage_done = all((m, t, "post") in done for t in range(m + 1))
        if not ckpt.exists() and not stage_done:
            out_dir = root / f"stage{m}"
            if out_dir.exists() and not a.dry_run:
                # unfinished earlier attempt: lerobot refuses an existing dir, so move it aside
                out_dir.rename(root / f"stage{m}_incomplete_{os.getpid()}")
            cmd = [*train_prefix, "--dataset.repo_id=lerobot/libero",
                   f"--dataset.episodes=[{','.join(map(str, eps))}]",
                   f"--output_dir={out_dir}", f"--seed={a.seed}", *POLICY_FLAGS,
                   f"--batch_size={a.batch_size}", f"--steps={a.steps}",
                   f"--save_freq={a.steps}", "--log_freq=100", "--wandb.enable=false"]
            rc = run(cmd, root / "logs" / f"train_stage{m}.log", a.dry_run, run_id)
            if rc != 0:
                raise SystemExit(f"training failed at stage {m}: see {root}/logs/train_stage{m}.log")

        # post cells: every task seen so far
        for t in range(m + 1):
            if (m, t, "post") not in done:
                n, k = evaluate(ckpt, t, f"post_s{m}_t{t}")
                record(m, t, "post", n, k)

    # text triangle
    print("\nsuccess rate (rows = stage, cols = task), %")
    for m in range(n_tasks):
        row = []
        for t in range(n_tasks):
            r = done.get((m, t, "post"))
            row.append(f"{100 * r['successes'] / r['n_rollouts']:5.0f}" if r and r["n_rollouts"] else "    .")
        print(f"stage {m}: " + " ".join(row))
    print(f"\nraw cells: {cells_path}")


if __name__ == "__main__":
    sys.exit(main())
