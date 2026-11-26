#!/usr/bin/env python
"""Continual FlowDPG on a LIBERO task sequence: the Overview's loop, with an online phase per task.

For every task m of --order (one "stage"):
  1. entry eval : the previous stage's policy on task m, before any training on it (zero-shot)
  2. add data   : task m's demos join the dataset for good
  3. offline    : train on the method's data scope (see METHODS), warm or cold (see below)
  4. online     : --rounds times: collect --collect_episodes rollouts of task m with the current
                  policy (failures kept), add them for good, keep training --online_steps,
                  evaluate task m (--round_eval_episodes). Stops early once --target is reached.
  5. checkpoint : the stage's final policy + critic + optimizer, plus stage<m>/manifest.json
  6. post eval  : the final policy on every task seen so far -> one row of the CLARE triangle

  python continual.py --method fdpg_cumulative --suite libero_spatial --order 0 1 2 --seed 0
  python report.py $RUNS/<run_id>/cells.jsonl          # triangle + AUC / FWT / NBT

Methods (the Overview's five comparisons; all get the same steps, rounds and episodes per stage):
  fdpg_cumulative  main method: warm start (policy, critic AND optimizer state) from the previous
                   stage, data = demos of tasks 1..m + every rollout collected so far
  bc_cumulative    same data and warm start, actor is plain BC (critic trained but never used)
  fdpg_current     warm start, data = demos + rollouts of the current task only
  fdpg_cold        every stage restarts from the same initialisation (same seed), cumulative data;
                   warm within the stage's online rounds
  fixed            trains stage 1 like the others, then never updates (only evaluates)

Design decisions (documented here because they change results):
  * The actor learns only from demo frames (merged datasets put all demos first, and
    --fdpg.actor_max_index = number of demo frames); rollouts train only the critic. Training the
    actor on its own rollouts collapsed on Spatial task 0 (see README).
  * Sampling is uniform over all frames of the stage's dataset (LeRobot's default).
  * Normalisation is fixed for the whole run: min/max stats of the demos of every task in --order,
    computed once. Without this, each warm start would rescale states and actions under the
    weights' feet, and the run would depend on which tasks happened to arrive first.
  * Evaluation uses env seed 0 for every cell (same starting positions for every method and run
    seed); collection uses seeds >= 1,000,000, so no evaluated start was ever trained on.
  * Each stage starts a fresh learning-rate schedule; only the optimizer's per-parameter state
    (Adam moments) is carried over (warm_start.py).

Saving GPU time and sharing the GPU:
  --stage0_from <run dir>  reuse stage 0 of a finished run with the same settings (its stage 0 is
                           identical for fixed, fdpg_cold, fdpg_current and fdpg_cumulative). Only
                           its records are copied; checkpoints and rollouts are read in place.
  --avoid "Mon,Thu,Fri 14:00-17:00"  (default) never run a job that would overlap these hours;
                           the runner sleeps until the window ends. --avoid none disables it.

Restartable: finished trainings, collections and evaluations are skipped on re-run.
Disk: intermediate round checkpoints and merged datasets are deleted once they are no longer
needed (--keep_all keeps them); the final checkpoint of every stage is kept.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "flowdpg_launch.py"
RES = ["--env.observation_height=256", "--env.observation_width=256"]  # must match the demos
POLICY = ["--policy.type=multi_task_dit", "--policy.objective=flow_matching",
          "--policy.num_integration_steps=10", "--policy.integration_method=euler",
          "--policy.num_layers=4", "--policy.hidden_dim=512", "--policy.num_heads=8"]
RESULT_RE = re.compile(r"'n_episodes': (\d+), 'n_success': (\d+)")
PEAK_RE = re.compile(r"peak_gpu_mem_gb=([\d.]+)")

# method -> (FlowDPG mode, warm start across stages, data scope)
METHODS = {
    "fdpg_cumulative": ("flowdpg", True, "cumulative"),
    "bc_cumulative": ("bc_critic", True, "cumulative"),
    "fdpg_current": ("flowdpg", True, "current"),
    "fdpg_cold": ("flowdpg", False, "cumulative"),
    "fixed": ("flowdpg", True, "cumulative"),   # stage 0 only
}


def info(ds: Path) -> dict:
    return json.loads((ds / "meta" / "info.json").read_text())


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def collect_seed(seed: int, stage: int, rnd: int) -> int:
    """Disjoint env seed ranges per (run seed, stage, round), all far away from eval seed 0."""
    return 1_000_000 * (seed + 1) + 10_000 * stage + 1_000 * rnd


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# rough job lengths on fafnir's RTX 4090 (pilot: 5k steps = 2,600 s), used only to avoid the windows
SEC_PER_STEP = 0.53


def parse_windows(spec: str) -> list[tuple[int, int, int]]:
    """'Mon,Thu 14:00-17:00; Fri 09:00-10:00' -> [(weekday, start_min, end_min), ...]."""
    out = []
    if not spec or spec.strip().lower() == "none":
        return out
    for part in spec.split(";"):
        days, _, hours = part.strip().partition(" ")
        lo, _, hi = hours.strip().partition("-")
        start = int(lo.split(":")[0]) * 60 + int(lo.split(":")[1])
        end = int(hi.split(":")[0]) * 60 + int(hi.split(":")[1])
        if end <= start:
            raise ValueError(f"bad window {part!r}: end must be after start")
        for d in days.split(","):
            out.append((DAYS.index(d.strip()[:3].capitalize()), start, end))
    return out


def blocked_until(now: datetime, expected_s: float, windows) -> datetime | None:
    """If a job starting now and lasting expected_s would overlap a window, the window's end."""
    job_end = now + timedelta(seconds=expected_s)
    for day_offset in range(0, 3):
        day = (now + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        for wd, start, end in windows:
            if day.weekday() != wd:
                continue
            w0, w1 = day + timedelta(minutes=start), day + timedelta(minutes=end)
            if now < w1 and job_end > w0:
                return w1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--order", type=int, nargs="+", required=True, help="env task ids in arrival order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.environ.get("DATA", "."),
                    help="where the per-task demo datasets are built once and shared by all runs")
    ap.add_argument("--out", default=None, help="runs folder (default: $DATA/continual)")
    ap.add_argument("--tag", default="", help="appended to the run id")
    ap.add_argument("--offline_steps", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=4, help="max online rounds per task")
    ap.add_argument("--online_steps", type=int, default=5000)
    ap.add_argument("--collect_episodes", type=int, default=50)
    ap.add_argument("--round_eval_episodes", type=int, default=10,
                    help="task-m eval after each round (0 = skip; then --target cannot stop early)")
    ap.add_argument("--target", type=float, default=None,
                    help="stop a task's online phase once its round eval reaches this success (0-1)")
    ap.add_argument("--eval_episodes", type=int, default=10, help="episodes per triangle cell")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--env_batch", type=int, default=10)
    ap.add_argument("--timestep", default="uniform", choices=["uniform", "beta"])
    ap.add_argument("--fdpg", nargs="*", default=[], metavar="KEY=VALUE", help="extra FlowDPG settings")
    ap.add_argument("--keep_all", action="store_true", help="keep round checkpoints and merged datasets")
    ap.add_argument("--stage0_from", default=None, help="reuse stage 0 of this finished run directory")
    ap.add_argument("--avoid", default="Mon,Thu,Fri 14:00-17:00",
                    help="hours when no job may run (GPU reserved for others); 'none' to disable")
    ap.add_argument("--dry_run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    for n in (a.collect_episodes, a.eval_episodes, a.round_eval_episodes):
        if n % a.env_batch:
            raise SystemExit("episode counts must be multiples of --env_batch")
    if len(set(a.order)) != len(a.order):
        raise SystemExit("--order has repeated tasks")
    if a.target is not None and a.round_eval_episodes == 0:
        raise SystemExit("--target needs --round_eval_episodes > 0")

    windows = parse_windows(a.avoid)
    mode, warm, scope = METHODS[a.method]
    order_s = "-".join(map(str, a.order))
    run_id = f"{a.method}_{a.suite}_o{order_s}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    root = Path(a.out or Path(os.environ.get("DATA", ".")) / "continual") / run_id
    py = sys.executable
    dry = a.dry_run

    def jsonl(name):
        p = root / name
        return [json.loads(line) for line in open(p)] if p.exists() else []

    def append(name, rec):
        print("RECORD", name, json.dumps(rec), flush=True)
        if not dry:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / name, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def run(cmd, log: Path) -> str:
        cmd = [str(c) for c in cmd]
        print(">>", " ".join(cmd), flush=True)
        if dry:
            return ""
        log.parent.mkdir(parents=True, exist_ok=True)
        local = Path(f"/tmp/{os.environ.get('USER', 'user')}_cfdpg_logs") / f"{run_id}__{log.name}"
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"   live log: {local}", flush=True)
        env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1"}
        with open(local, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        shutil.copy(local, log)
        if rc != 0:
            raise SystemExit(f"failed (exit {rc}): see {log}")
        return log.read_text()

    def wait_for_gpu(expected_s: float, what: str):
        """Sleep while a job of this length would overlap a reserved window."""
        while not dry:
            until = blocked_until(datetime.now(), expected_s, windows)
            if until is None:
                return
            print(f"   {datetime.now():%a %H:%M}: {what} (~{expected_s / 60:.0f} min) would overlap the "
                  f"reserved GPU hours; waiting until {until:%a %H:%M}", flush=True)
            time.sleep(max(60.0, (until - datetime.now()).total_seconds() + 60))

    def rm(p: Path):
        if not a.keep_all and p.exists() and not dry:
            shutil.rmtree(p)

    # ---------------------------------------------------------------- demos and fixed statistics
    set_dir = Path(a.data_root) / f"{a.suite}_tasks{'-'.join(map(str, sorted(a.order)))}"
    all_demos = set_dir / "all_demos"
    if not (all_demos / "meta" / "info.json").exists():
        if all_demos.exists() and not dry:
            shutil.rmtree(all_demos)
        run([py, HERE / "build_task_dataset.py", "--suite", a.suite, "--task_ids", *sorted(a.order),
             "--out", all_demos], set_dir / "logs" / "build_all.log")
    stats_file = all_demos / "meta" / "stats.json"

    def fix_stats(ds: Path):
        """Overwrite a dataset's stats with the run's fixed stats (original kept as stats_own.json)."""
        if dry:
            return
        own, st = ds / "meta" / "stats_own.json", ds / "meta" / "stats.json"
        if not own.exists():
            shutil.copy(st, own)
        if st.read_bytes() != stats_file.read_bytes():
            shutil.copy(stats_file, st)

    def demos_of(task_id: int) -> Path:
        d = set_dir / f"{a.suite}_t{task_id}_demos"
        if not (d / "meta" / "info.json").exists():
            if d.exists() and not dry:
                shutil.rmtree(d)
            run([py, HERE / "build_task_dataset.py", "--suite", a.suite, "--task_ids", task_id,
                 "--out", d], set_dir / "logs" / f"build_t{task_id}.log")
        fix_stats(d)
        return d

    def frames(ds: Path) -> int:
        return 1000 if dry else int(info(ds)["total_frames"])

    # ---------------------------------------------------------------- building blocks
    evals = {(e["ckpt"], e["task_id"], e["n"]): e for e in jsonl("evals.jsonl")}

    def evaluate(ckpt: Path, task_id: int, n: int, tag: str) -> tuple[int, int]:
        """Cached: a checkpoint is evaluated on a task once (eval seed 0 is deterministic)."""
        key = (str(ckpt), task_id, n)
        if key in evals:
            return evals[key]["n"], evals[key]["k"]
        wait_for_gpu(120 + 30 * n, f"eval {tag}")
        text = run([py, LAUNCHER, "eval", f"--policy.path={ckpt}", "--env.type=libero",
                    f"--env.task={a.suite}", f"--env.task_ids=[{task_id}]", *RES,
                    f"--eval.batch_size={a.env_batch}", f"--eval.n_episodes={n}", "--seed=0",
                    f"--output_dir={root / 'eval' / tag}"], root / "logs" / f"eval_{tag}.log")
        if dry:
            return n, 0
        m = RESULT_RE.findall(text)
        if not m:
            raise SystemExit(f"could not parse the eval result of {tag}")
        rec = {"ckpt": str(ckpt), "task_id": task_id, "n": int(m[-1][0]), "k": int(m[-1][1])}
        evals[key] = rec
        append("evals.jsonl", rec)
        return rec["n"], rec["k"]

    def ckpt_of(train_dir: Path, steps: int) -> Path:
        return train_dir / "checkpoints" / f"{steps:06d}" / "pretrained_model"

    def train(data: Path, train_dir: Path, steps: int, init: Path | None, n_demo_frames: int,
              tag: str) -> dict:
        """One lerobot-train call. init=None: fresh policy. Otherwise weights AND optimizer state
        come from `init`. Returns time and memory for the manifest."""
        side = train_dir.with_name(train_dir.name + "_cost.json")
        if ckpt_of(train_dir, steps).exists():
            return json.loads(side.read_text()) if side.exists() else {}
        if train_dir.exists() and not dry:
            shutil.rmtree(train_dir)            # an interrupted run: lerobot refuses an existing dir
        cmd = [py, LAUNCHER, "train", f"--fdpg.mode={mode}", *[f"--fdpg.{kv}" for kv in a.fdpg],
               f"--fdpg.actor_max_index={n_demo_frames}"]
        if init is None:
            cmd += POLICY + [f"--policy.timestep_sampling_strategy={a.timestep}"]
        else:
            cmd += [f"--policy.path={init}", f"--warm.optimizer={init.parent}"]
        cmd += [f"--dataset.repo_id=local/{data.name}", f"--dataset.root={data}",
                f"--output_dir={train_dir}", f"--seed={a.seed}", "--policy.device=cuda",
                "--policy.push_to_hub=false", f"--batch_size={a.batch_size}", f"--steps={steps}",
                f"--save_freq={steps}", "--log_freq=100", "--wandb.enable=false"]
        wait_for_gpu(300 + SEC_PER_STEP * steps, f"training {tag}")
        t0 = time.time()
        text = run(cmd, root / "logs" / f"train_{tag}.log")
        peak = PEAK_RE.findall(text)
        cost = {"wall_s": round(time.time() - t0), "peak_gpu_mem_gb": float(peak[-1]) if peak else None}
        if not dry:
            side.write_text(json.dumps(cost))
        return cost

    def collect(ckpt: Path, task_id: int, out: Path, stage: int, rnd: int) -> dict:
        f = out / "episodes.jsonl"
        n_done = sum(1 for _ in open(f)) if f.exists() else 0
        if n_done != a.collect_episodes:        # interrupted collections are redone completely
            if out.exists() and not dry:
                shutil.rmtree(out)
            wait_for_gpu(120 + 30 * a.collect_episodes, f"collection s{stage} r{rnd}")
            run([py, HERE / "collect_rollouts.py", f"--policy_path={ckpt}", f"--suite={a.suite}",
                 f"--task_id={task_id}", f"--n_episodes={a.collect_episodes}",
                 f"--batch_size={a.env_batch}", f"--seed={collect_seed(a.seed, stage, rnd)}",
                 f"--out={out}"], root / "logs" / f"collect_s{stage}_r{rnd}.log")
        if dry:
            return {"collect_success": 0, "collect_episodes": a.collect_episodes}
        eps = [json.loads(line) for line in open(f)]
        return {"collect_success": sum(e["success"] for e in eps), "collect_episodes": len(eps)}

    def dataset(demos: list[Path], rollouts: list[Path], out: Path) -> Path:
        """Demos first (the actor's frames), then rollouts. One input is used as it is."""
        inputs = demos + rollouts
        if len(inputs) == 1:
            return inputs[0]
        if not (out / "meta" / "info.json").exists():
            if out.exists() and not dry:
                shutil.rmtree(out)
            run([py, HERE / "merge_data.py", "--inputs", *inputs, "--out", out],
                root / "logs" / f"merge_{out.parent.name}_{out.name}.log")
        fix_stats(out)
        return out

    def describe(ds: Path) -> dict:
        if dry:
            return {"path": str(ds)}
        i = info(ds)
        return {"path": str(ds), "episodes": i["total_episodes"], "frames": i["total_frames"]}

    # ---------------------------------------------------------------- the stages
    cells = {(c["stage"], c["task"], c["kind"]): c for c in jsonl("cells.jsonl")}
    stages = {s["stage"]: s for s in jsonl("stages.jsonl")}
    # Rounds already finished (their checkpoints/data may be cleaned up): never redone.
    rounds_done = {(r["stage"], r["round"]): r for r in jsonl("online.jsonl")}
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps({**vars(a), "run_id": run_id, "fdpg_mode": mode,
                                                      "warm": warm, "scope": scope,
                                                      "git_commit": git_commit()}, indent=2))

    def reuse_stage0(src: Path):
        """Copy stage 0's records from a finished run with identical stage-0 settings."""
        cfg = json.loads((src / "config.json").read_text())
        same = ["suite", "seed", "offline_steps", "rounds", "online_steps", "collect_episodes",
                "round_eval_episodes", "target", "eval_episodes", "batch_size", "timestep", "fdpg"]
        diff = [k for k in same if cfg.get(k) != getattr(a, k)]
        if cfg.get("fdpg_mode") != mode:
            diff.append("fdpg mode")
        if sorted(cfg["order"]) != sorted(a.order) or cfg["order"][0] != a.order[0]:
            diff.append("order (same first task and same task set needed)")
        if diff:
            raise SystemExit(f"--stage0_from {src}: settings differ ({', '.join(diff)})")
        src_stage = next((r for r in map(json.loads, open(src / "stages.jsonl")) if r["stage"] == 0), None)
        if src_stage is None:
            raise SystemExit(f"--stage0_from {src}: stage 0 is not finished there")
        final = Path(src_stage["final_ckpt"])
        if warm and a.method != "fixed" and len(a.order) > 1 and not (final.parent / "training_state").exists():
            raise SystemExit(f"--stage0_from {src}: its stage-0 optimizer state is gone (deleted after its "
                             f"stage 1), and {a.method} warm-starts from it. Run without --stage0_from.")
        under = str(src / "stage0")
        for e in map(json.loads, open(src / "evals.jsonl")):
            if e["ckpt"].startswith(under) and (e["ckpt"], e["task_id"], e["n"]) not in evals:
                evals[(e["ckpt"], e["task_id"], e["n"])] = e
                append("evals.jsonl", e)
        for r in map(json.loads, open(src / "online.jsonl")):
            if r["stage"] == 0:
                rounds_done[(0, r["round"])] = r
                append("online.jsonl", r)
        for c in map(json.loads, open(src / "cells.jsonl")):
            if c["stage"] == 0:
                cell(0, c["task"], c["kind"], (c["n_rollouts"], c["successes"]))
        rec = {**src_stage, "method": a.method, "run_id": run_id, "reused_from": str(src)}
        (root / "stage0").mkdir(parents=True, exist_ok=True)
        (root / "stage0" / "manifest.json").write_text(json.dumps(rec, indent=2))
        stages[0] = rec
        append("stages.jsonl", rec)
        print(f"   stage 0 reused from {src}", flush=True)

    def cell(stage, pos, kind, nk):
        if (stage, pos, kind) in cells:
            return
        rec = dict(run_id=run_id, method=a.method, seed=a.seed, order=f"o{order_s}", stage=stage,
                   task=pos, kind=kind, n_rollouts=nk[0], successes=nk[1])
        cells[(stage, pos, kind)] = rec
        append("cells.jsonl", rec)

    if a.stage0_from and 0 not in stages and not dry:
        reuse_stage0(Path(a.stage0_from))

    all_rollouts: list[Path] = []     # every rollout dataset of this run, in collection order
    prev_final: Path | None = None
    total_steps = 0
    for m, task_id in enumerate(a.order):
        sdir = root / f"stage{m}"
        if m in stages:                                        # finished on an earlier run
            s = stages[m]
            prev_final, total_steps = Path(s["final_ckpt"]), s["gradient_steps_total"]
            all_rollouts += [Path(p) for p in s["rollouts"]]
            continue

        # 1. entry: zero-shot on the new task
        if m >= 1:
            cell(m, m, "entry", evaluate(prev_final, task_id, a.eval_episodes, f"entry_s{m}_t{task_id}"))

        record = {"stage": m, "task_id": task_id, "method": a.method, "run_id": run_id,
                  "init_from": None, "rounds": [], "rollouts": [], "fdpg": a.fdpg,
                  "stats_file": str(stats_file),
                  "stats_hash": None if dry else file_hash(stats_file)}
        steps_stage, walls, peaks = 0, 0, []

        if a.method == "fixed" and m >= 1:
            final = prev_final
            record["note"] = "fixed initial policy: no training"
        else:
            # 2.-3. offline on the stage's data
            demos = [demos_of(t) for t in (a.order[: m + 1] if scope == "cumulative" else [task_id])]
            old_rollouts = list(all_rollouts) if scope == "cumulative" else []
            n_demo = sum(frames(d) for d in demos)
            init = prev_final if (warm and m >= 1) else None
            record["init_from"] = str(init) if init else "fresh (seed %d)" % a.seed
            if (m, 0) in rounds_done:
                rinfo = {k: v for k, v in rounds_done[(m, 0)].items() if k != "stage"}
            else:
                data = dataset(demos, old_rollouts, sdir / "data_r0")
                tdir = sdir / "train_r0"
                cost = train(data, tdir, a.offline_steps, init, n_demo, f"s{m}_r0")
                rinfo = {"round": 0, "data": describe(data), "train_steps": a.offline_steps,
                         "ckpt": str(ckpt_of(tdir, a.offline_steps)), **cost}
                if data.parent == sdir:
                    rm(data)
                if (init is not None and not a.keep_all and not dry  # the old optimizer state is used up
                        and root in init.parents):                   # (never another run's files)
                    shutil.rmtree(init.parent / "training_state", ignore_errors=True)
                if a.round_eval_episodes:
                    n, k = evaluate(Path(rinfo["ckpt"]), task_id, a.round_eval_episodes, f"round_s{m}_r0")
                    rinfo["eval"] = [n, k]
                append("online.jsonl", {"stage": m, **rinfo})
            ckpt = Path(rinfo["ckpt"])
            steps_stage += rinfo["train_steps"]
            walls += rinfo.get("wall_s") or 0
            peaks.append(rinfo.get("peak_gpu_mem_gb"))
            record["rounds"].append(rinfo)

            # 4. online rounds on task m
            new_rollouts: list[Path] = []
            for r in range(1, a.rounds + 1):
                last = record["rounds"][-1]
                if a.target is not None and "eval" in last and last["eval"][1] >= a.target * last["eval"][0]:
                    print(f"   stage {m}: target {a.target} reached after round {r - 1}", flush=True)
                    break
                roll = sdir / f"rollouts_r{r}"
                new_rollouts.append(roll)
                if (m, r) in rounds_done:
                    rinfo = {k: v for k, v in rounds_done[(m, r)].items() if k != "stage"}
                else:
                    cstats = collect(ckpt, task_id, roll, m, r)
                    data = dataset(demos, old_rollouts + new_rollouts, sdir / f"data_r{r}")
                    tdir = sdir / f"train_r{r}"
                    cost = train(data, tdir, a.online_steps, ckpt, n_demo, f"s{m}_r{r}")
                    rinfo = {"round": r, **cstats, "rollouts": str(roll), "data": describe(data),
                             "train_steps": a.online_steps, "ckpt": str(ckpt_of(tdir, a.online_steps)),
                             **cost}
                    rm(data)
                    rm(ckpt.parents[2])                # previous round: it has collected and been trained on
                    if a.round_eval_episodes:
                        n, k = evaluate(Path(rinfo["ckpt"]), task_id, a.round_eval_episodes,
                                        f"round_s{m}_r{r}")
                        rinfo["eval"] = [n, k]
                    append("online.jsonl", {"stage": m, **rinfo})
                ckpt = Path(rinfo["ckpt"])
                steps_stage += rinfo["train_steps"]
                walls += rinfo.get("wall_s") or 0
                peaks.append(rinfo.get("peak_gpu_mem_gb"))
                record["rounds"].append(rinfo)
            final = ckpt
            all_rollouts += new_rollouts
            record["rollouts"] = [str(p) for p in new_rollouts]
            record["actor_frames"] = n_demo

        # 5. checkpoint bookkeeping
        total_steps += steps_stage
        known = [p for p in peaks if p is not None]
        record.update({"final_ckpt": str(final), "gradient_steps_stage": steps_stage,
                       "gradient_steps_total": total_steps, "train_wall_s": walls,
                       "peak_gpu_mem_gb": max(known) if known else None,
                       "rollouts_total": [str(p) for p in all_rollouts]})

        # 6. post: every task so far
        for pos, t in enumerate(a.order[: m + 1]):
            cell(m, pos, "post", evaluate(final, t, a.eval_episodes, f"post_s{m}_t{t}"))

        if not dry:
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "manifest.json").write_text(json.dumps(record, indent=2))
        stages[m] = record
        append("stages.jsonl", record)
        prev_final = final

    # ---------------------------------------------------------------- summary
    n = len(a.order)
    print(f"\n{run_id}: success per cell, % (rows = stage, cols = task position; e = entry)")
    for m in range(n):
        row = []
        for t in range(n):
            c = cells.get((m, t, "post"))
            row.append(f"{100 * c['successes'] / c['n_rollouts']:4.0f}" if c and c["n_rollouts"] else "   .")
        e = cells.get((m, m, "entry"))
        tail = f"   e {100 * e['successes'] / e['n_rollouts']:.0f}" if e and e["n_rollouts"] else ""
        print(f"stage {m}: " + " ".join(row) + tail)
    print(f"cells: {root / 'cells.jsonl'}   ->   python {HERE / 'report.py'} {root / 'cells.jsonl'}")


if __name__ == "__main__":
    main()
