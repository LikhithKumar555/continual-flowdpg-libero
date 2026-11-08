#!/usr/bin/env python
"""Single-task offline -> online FlowDPG loop (the supervisor's milestone).

  round 0  offline: FlowDPG from scratch on the task's demos (build_task_dataset.py output)
  round r  online:  collect rollouts with the round r-1 policy (all episodes kept, failures too)
                    -> merge demos + every rollout round so far
                    -> continue training the round r-1 policy and critic (warm start) on that data
                    -> evaluate on fixed starting positions (eval seed 0, never used for collection)

  python online_loop.py --suite libero_spatial --task_id 0 --demos $DATA/spatial_t0_demos \
      --out $DATA/online_spatial_t0_s0 --seed 0 --rounds 4

Everything lands in --out (checkpoints, datasets, logs) plus progress.jsonl with one line per
round: eval success, collection success, data size. Restartable: finished steps are skipped.
Evaluation and collection use disjoint seed ranges, so the reported success is never measured
on starting positions the policy was trained on.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "flowdpg_launch.py"
RES = ["--env.observation_height=256", "--env.observation_width=256"]
POLICY = ["--policy.type=multi_task_dit", "--policy.objective=flow_matching",
          "--policy.num_integration_steps=10", "--policy.integration_method=euler",
          "--policy.num_layers=4", "--policy.hidden_dim=512", "--policy.num_heads=8"]
RESULT_RE = re.compile(r"'n_episodes': (\d+), 'n_success': (\d+)")


def run(cmd, log: Path, dry: bool) -> None:
    print(">>", " ".join(map(str, cmd)), flush=True)
    if dry:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    local = Path(f"/tmp/{os.environ.get('USER', 'user')}_cfdpg_logs") / f"online_{log.parent.parent.name}_{log.name}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"   live log: {local}", flush=True)
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1"}
    with open(local, "w") as f:
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    shutil.copy(local, log)
    if rc != 0:
        raise SystemExit(f"failed (exit {rc}): see {log}")


def stats_shift(ref: Path, new: Path) -> str:
    """Largest relative change of the min/max range of state and action between two stats files,
    in units of the reference range (0 = identical scaling)."""
    a, b = json.loads(Path(ref).read_text()), json.loads(Path(new).read_text())
    parts = []
    for key in ("observation.state", "action"):
        if key not in a or key not in b or "min" not in a[key]:
            continue
        lo_a, hi_a = a[key]["min"], a[key]["max"]
        lo_b, hi_b = b[key]["min"], b[key]["max"]
        worst = 0.0
        for la, ha, lb, hb in zip(lo_a, hi_a, lo_b, hi_b):
            rng = max(ha - la, 1e-8)
            worst = max(worst, abs(lb - la) / rng, abs(hb - ha) / rng)
        parts.append(f"{key} up to {100 * worst:.0f}% of its range")
    return "; ".join(parts) or "n/a"


def ckpt_of(train_dir: Path, steps: int) -> Path:
    return train_dir / "checkpoints" / f"{steps:06d}" / "pretrained_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--demos", required=True, help="dataset from build_task_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4, help="online rounds after the offline round")
    ap.add_argument("--offline_steps", type=int, default=10000)
    ap.add_argument("--online_steps", type=int, default=5000, help="training steps per online round")
    ap.add_argument("--collect_episodes", type=int, default=50)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--env_batch", type=int, default=10)
    ap.add_argument("--timestep", default="uniform", choices=["uniform", "beta"])
    ap.add_argument("--fdpg", nargs="*", default=[], metavar="KEY=VALUE",
                    help="FlowDPG settings for the offline round, e.g. alpha=0.25 (kept by warm starts)")
    ap.add_argument("--extra_rollouts", nargs="*", default=[],
                    help="already collected rollout datasets to include from round 1 on")
    ap.add_argument("--actor_demos_only", action="store_true",
                    help="the actor learns only from the demo frames (they come first in every merged "
                         "dataset); rollouts, failures included, train only the critic")
    ap.add_argument("--recompute_stats", action="store_true",
                    help="let each round renormalise with the merged data's stats (old behaviour; "
                         "default keeps the demo stats fixed)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.collect_episodes % a.env_batch or a.eval_episodes % a.env_batch:
        raise SystemExit("--collect_episodes and --eval_episodes must be multiples of --env_batch")

    out = Path(a.out)
    demos = Path(a.demos)
    progress = out / "progress.jsonl"
    done = {}
    if progress.exists():
        for line in open(progress):
            r = json.loads(line)
            done[r["round"]] = r
    py = sys.executable

    def demo_frames() -> int:
        if a.dry_run and not (demos / "meta" / "info.json").exists():
            return 4487
        return int(json.loads((demos / "meta" / "info.json").read_text())["total_frames"])

    def train(data_dir: Path, train_dir: Path, steps: int, init: Path | None, tag: str):
        if ckpt_of(train_dir, steps).exists():
            return
        if train_dir.exists() and not a.dry_run:
            train_dir.rename(train_dir.with_name(train_dir.name + f"_incomplete_{os.getpid()}"))
        # --fdpg settings apply to every round (warm starts too); the critic architecture of a warm
        # start is read from its checkpoint by the launcher.
        cmd = [py, LAUNCHER, "train"] + [f"--fdpg.{kv}" for kv in a.fdpg]
        if a.actor_demos_only:
            cmd.append(f"--fdpg.actor_max_index={demo_frames()}")
        if init is None:
            cmd += POLICY + [f"--policy.timestep_sampling_strategy={a.timestep}"]
        else:
            cmd += [f"--policy.path={init}"]
        cmd += [f"--dataset.repo_id=local/{data_dir.name}", f"--dataset.root={data_dir}",
                f"--output_dir={train_dir}", f"--seed={a.seed}", "--policy.device=cuda",
                "--policy.push_to_hub=false", f"--batch_size={a.batch_size}", f"--steps={steps}",
                f"--save_freq={steps}", "--log_freq=100", "--wandb.enable=false"]
        run(cmd, out / "logs" / f"train_{tag}.log", a.dry_run)

    def evaluate(ckpt: Path, tag: str) -> tuple[int, int]:
        log = out / "logs" / f"eval_{tag}.log"
        run([py, LAUNCHER, "eval", f"--policy.path={ckpt}", "--env.type=libero", f"--env.task={a.suite}",
             f"--env.task_ids=[{a.task_id}]", *RES, f"--eval.batch_size={a.env_batch}",
             f"--eval.n_episodes={a.eval_episodes}", "--seed=0", f"--output_dir={out / 'eval' / tag}"],
            log, a.dry_run)
        if a.dry_run:
            return 0, 0
        m = RESULT_RE.findall(log.read_text())
        if not m:
            raise SystemExit(f"could not parse eval result: {log}")
        return int(m[-1][0]), int(m[-1][1])

    def n_collected(data_dir: Path) -> int:
        f = data_dir / "episodes.jsonl"
        return sum(1 for _ in open(f)) if f.exists() else 0

    def collect(ckpt: Path, data_dir: Path, r: int) -> dict:
        # An interrupted collection still writes episodes.jsonl (for the episodes it finished),
        # so only a full set counts as done; anything else is deleted and collected again.
        if n_collected(data_dir) != a.collect_episodes:
            if data_dir.exists() and not a.dry_run:
                print(f"   discarding incomplete collection {data_dir} ({n_collected(data_dir)} episodes)", flush=True)
                shutil.rmtree(data_dir)
            merged = out / f"data_r{r}"
            if merged.exists() and not a.dry_run:      # built from the incomplete collection
                shutil.rmtree(merged)
            run([py, HERE / "collect_rollouts.py", f"--policy_path={ckpt}", f"--suite={a.suite}",
                 f"--task_id={a.task_id}", f"--n_episodes={a.collect_episodes}", f"--batch_size={a.env_batch}",
                 f"--seed={100000 * (a.seed + 1) + 1000 * r}", f"--out={data_dir}"],
                out / "logs" / f"collect_r{r}.log", a.dry_run)
        if a.dry_run:
            return {"collect_success": 0, "collect_episodes": 0}
        eps = [json.loads(line) for line in open(data_dir / "episodes.jsonl")]
        return {"collect_success": sum(e["success"] for e in eps), "collect_episodes": len(eps)}

    def merge(inputs: list[Path], data_dir: Path, r: int):
        if not (data_dir / "meta" / "info.json").exists():
            if data_dir.exists() and not a.dry_run:
                shutil.rmtree(data_dir)
            run([py, HERE / "merge_data.py", "--inputs", *inputs, "--out", data_dir],
                out / "logs" / f"merge_r{r}.log", a.dry_run)
        if a.recompute_stats or a.dry_run:
            return
        # Warm-started lerobot-train replaces the checkpoint's normalisation with the training
        # dataset's stats. Keep the round-0 (demo) stats so the weights see the same scaling.
        stats, demo_stats = data_dir / "meta" / "stats.json", demos / "meta" / "stats.json"
        backup = data_dir / "meta" / "stats_merged.json"
        if not backup.exists():
            shutil.copy(stats, backup)
            print("   normalisation shift the merge would have caused:", stats_shift(demo_stats, backup), flush=True)
        shutil.copy(demo_stats, stats)

    def record(rec: dict):
        done[rec["round"]] = rec
        print("RESULT", json.dumps(rec), flush=True)
        if not a.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            with open(progress, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ---- round 0: offline on demos
    t0 = out / "train_r0"
    train(demos, t0, a.offline_steps, None, "r0")
    ckpt = ckpt_of(t0, a.offline_steps)
    if 0 not in done:
        n, k = evaluate(ckpt, "r0")
        record({"round": 0, "eval_success": k, "eval_episodes": n, "data": str(demos),
                "train_steps_total": a.offline_steps})

    rollouts = [Path(p) for p in a.extra_rollouts]
    for r in range(1, a.rounds + 1):
        roll = out / f"rollouts_r{r}"
        stats = collect(ckpt, roll, r)
        rollouts.append(roll)
        data = out / f"data_r{r}"
        merge([demos, *rollouts], data, r)
        tr = out / f"train_r{r}"
        train(data, tr, a.online_steps, ckpt, f"r{r}")
        ckpt = ckpt_of(tr, a.online_steps)
        if r not in done:
            n, k = evaluate(ckpt, f"r{r}")
            record({"round": r, "eval_success": k, "eval_episodes": n, **stats,
                    "data": str(data), "rollout_datasets": len(rollouts),
                    "train_steps_total": a.offline_steps + r * a.online_steps})

    print("\nround | eval success | collected by the previous policy")
    for r in sorted(done):
        d = done[r]
        col = f"{d['collect_success']}/{d['collect_episodes']}" if "collect_success" in d else "-"
        pct = 100 * d["eval_success"] / d["eval_episodes"] if d["eval_episodes"] else 0
        print(f"{r:5d} | {d['eval_success']:3d}/{d['eval_episodes']} ({pct:3.0f}%) | {col}")
    print(f"progress file: {progress}")


if __name__ == "__main__":
    main()
