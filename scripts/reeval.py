#!/usr/bin/env python
"""Re-evaluate finished continual runs with more episodes per triangle cell. Nothing is retrained.

continual.py keeps every stage's final checkpoint, so each cell of the triangle can be measured
again: post cells (stage m's final policy on every task so far) and entry cells (stage m-1's
final policy on task m, before training on it). The pilot used 10 episodes per cell (about +-15%);
50 episodes brings that to about +-7%.

  python reeval.py --episodes 50 $DATA/continual/fdpg_cumulative_libero_spatial_o0-1-2_s0 \
      $DATA/continual/fixed_libero_spatial_o0-1-2_s0 ...
  python report.py <run dir>/cells_e50.jsonl

Writes <run dir>/cells_e50.jsonl in the same format as cells.jsonl (report.py reads it as it is).
A checkpoint is evaluated once per task: runs that share checkpoints (stage 0 reused by fixed and
fdpg_cold) share the result. Evaluation uses env seed 0, like every other evaluation of the
project. Restartable, and it waits out the reserved GPU hours (--avoid, as in continual.py).
Do not run it while training runs on the same GPU: start it after the queue has finished.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from continual import LAUNCHER, RES, RESULT_RE, blocked_until, parse_windows  # noqa: E402


def plan(run: Path) -> list[dict]:
    """Every triangle cell of a finished run: which checkpoint on which task."""
    cfg = json.loads((run / "config.json").read_text())
    stages = {s["stage"]: s for s in map(json.loads, open(run / "stages.jsonl"))}
    order = cfg["order"]
    missing = [m for m in range(len(order)) if m not in stages]
    if missing:
        raise SystemExit(f"{run.name}: stages {missing} are not finished yet")
    cells = []
    for m in range(len(order)):
        if m >= 1:
            cells.append(dict(stage=m, task=m, kind="entry", ckpt=stages[m - 1]["final_ckpt"], task_id=order[m]))
        for t in range(m + 1):
            cells.append(dict(stage=m, task=t, kind="post", ckpt=stages[m]["final_ckpt"], task_id=order[t]))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories of continual.py")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--env_batch", type=int, default=10)
    ap.add_argument("--avoid", default="Mon,Thu,Fri 14:00-17:00")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.episodes % a.env_batch:
        raise SystemExit("--episodes must be a multiple of --env_batch")
    windows = parse_windows(a.avoid)
    runs = [Path(r) for r in a.runs]
    py = sys.executable

    # results of every run's evals.jsonl, keyed by (checkpoint, task, episodes): shared checkpoints
    # are evaluated once
    cache = {}
    for run in runs:
        f = run / "evals.jsonl"
        if f.exists():
            for e in map(json.loads, open(f)):
                cache[(e["ckpt"], e["task_id"], e["n"])] = e

    def evaluate(run: Path, ckpt: str, task_id: int, tag: str) -> dict:
        key = (ckpt, task_id, a.episodes)
        if key in cache:
            return cache[key]
        if not Path(ckpt).exists() and not a.dry_run:
            raise SystemExit(f"checkpoint missing: {ckpt}")
        cmd = [py, str(LAUNCHER), "eval", f"--policy.path={ckpt}", "--env.type=libero",
               f"--env.task={cfg_suite[run]}", f"--env.task_ids=[{task_id}]", *RES,
               f"--eval.batch_size={a.env_batch}", f"--eval.n_episodes={a.episodes}", "--seed=0",
               f"--output_dir={run / f'eval_e{a.episodes}' / tag}"]
        print(">>", " ".join(cmd), flush=True)
        if a.dry_run:
            rec = {"ckpt": ckpt, "task_id": task_id, "n": a.episodes, "k": 0}
            cache[key] = rec
            return rec
        expected = 120 + 30 * a.episodes
        while (until := blocked_until(datetime.now(), expected, windows)) is not None:
            print(f"   {datetime.now():%a %H:%M}: waiting for the reserved GPU hours to end ({until:%a %H:%M})",
                  flush=True)
            time.sleep(max(60.0, (until - datetime.now()).total_seconds() + 60))
        log = run / "logs" / f"eval_e{a.episodes}_{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        local = Path(f"/tmp/{os.environ.get('USER', 'user')}_cfdpg_logs") / f"{run.name}__{log.name}"
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"   live log: {local}", flush=True)
        env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1"}
        with open(local, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        shutil.copy(local, log)
        m = RESULT_RE.findall(log.read_text())
        if rc != 0 or not m:
            raise SystemExit(f"eval failed: see {log}")
        rec = {"ckpt": ckpt, "task_id": task_id, "n": int(m[-1][0]), "k": int(m[-1][1])}
        cache[key] = rec
        with open(run / "evals.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    cfg_suite = {}
    todo = []
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        cfg_suite[run] = cfg["suite"]
        todo.append((run, cfg, plan(run)))

    for run, cfg, cells in todo:
        out = run / f"cells_e{a.episodes}.jsonl"
        done = set()
        if out.exists():
            done = {(c["stage"], c["task"], c["kind"]) for c in map(json.loads, open(out))}
        for c in cells:
            if (c["stage"], c["task"], c["kind"]) in done:
                continue
            tag = f"{c['kind']}_s{c['stage']}_t{c['task_id']}"
            r = evaluate(run, c["ckpt"], c["task_id"], tag)
            rec = dict(run_id=run.name, method=cfg["method"], seed=cfg["seed"],
                       order="o" + "-".join(map(str, cfg["order"])), stage=c["stage"], task=c["task"],
                       kind=c["kind"], n_rollouts=r["n"], successes=r["k"])
            print("CELL", json.dumps(rec), flush=True)
            if not a.dry_run:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    if not a.dry_run:
        print(f"\nsuccess per cell, % ({a.episodes} episodes; rows = stage, cols = task; e = entry)")
        for run, cfg, cells in todo:
            recs = {(c["stage"], c["task"], c["kind"]): c
                    for c in map(json.loads, open(run / f"cells_e{a.episodes}.jsonl"))}
            n = len(cfg["order"])
            print(f"--- {run.name}")
            for m in range(n):
                row = [f"{100 * recs[(m, t, 'post')]['successes'] / recs[(m, t, 'post')]['n_rollouts']:4.0f}"
                       if t <= m else "   ." for t in range(n)]
                e = recs.get((m, m, "entry"))
                tail = f"   e {100 * e['successes'] / e['n_rollouts']:.0f}" if e else ""
                print(f"stage {m}: " + " ".join(row) + tail)


if __name__ == "__main__":
    main()
