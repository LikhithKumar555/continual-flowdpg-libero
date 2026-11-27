"""Optimizer warm start (warm_start.py) and the continual runner's plan (continual.py --dry_run)."""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import warm_start  # noqa: E402


def lerobot_save(opt, d: Path):
    """What LeRobot writes: per-parameter state flattened with '/', param groups as json."""
    sd = opt.state_dict()
    flat = {f"state/{i}/{k}": v.contiguous() for i, st in sd["state"].items() for k, v in st.items()}
    d.mkdir(parents=True)
    save_file(flat, str(d / warm_start.OPTIMIZER_STATE))


def model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def trained_adam(m, lr=1e-3, steps=5):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        m(torch.randn(16, 4)).pow(2).mean().backward()
        opt.step()
    return opt


def test_state_is_copied_and_param_groups_kept(tmp_path):
    m = model()
    old = trained_adam(m, lr=1e-3)
    ckpt = tmp_path / "010000"
    lerobot_save(old, ckpt / "training_state")

    new = torch.optim.AdamW(m.parameters(), lr=5e-4)
    n = warm_start.load_param_state(new, ckpt / "pretrained_model")   # pretrained_model path works too
    assert n == len(list(m.parameters()))
    for p in m.parameters():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(new.state[p][key], old.state[p][key])
    assert new.param_groups[0]["lr"] == 5e-4          # the new run's schedule, not the old lr


def test_shape_mismatch_is_an_error(tmp_path):
    lerobot_save(trained_adam(model()), tmp_path / "training_state")
    other = torch.nn.Sequential(torch.nn.Linear(4, 9), torch.nn.ReLU(), torch.nn.Linear(9, 2))
    with pytest.raises(ValueError, match="shape"):
        warm_start.load_param_state(torch.optim.AdamW(other.parameters()), tmp_path)


def test_more_saved_params_than_optimizer_is_an_error(tmp_path):
    lerobot_save(trained_adam(model()), tmp_path / "training_state")
    small = torch.nn.Linear(4, 8)
    with pytest.raises(ValueError):
        warm_start.load_param_state(torch.optim.AdamW(small.parameters()), tmp_path)


def test_missing_state_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        warm_start.training_state_dir(tmp_path)


def dry(method, tmp_path, *extra):
    out = subprocess.run([sys.executable, str(SCRIPTS / "continual.py"), "--method", method,
                          "--order", "3", "5", "--rounds", "2", "--data_root", str(tmp_path / "d"),
                          "--out", str(tmp_path / "runs"), "--dry_run", *extra],
                         capture_output=True, text=True, check=True).stdout
    return [line[3:] for line in out.splitlines() if line.startswith(">> ")]


def trains(cmds):
    return [c for c in cmds if " train " in c]


def test_main_method_warm_starts_everything_and_keeps_data(tmp_path):
    cmds = dry("fdpg_cumulative", tmp_path)
    tr = trains(cmds)
    assert len(tr) == 6                                   # 2 stages x (offline + 2 rounds)
    assert "--policy.type=multi_task_dit" in tr[0]        # only the very first training is fresh
    for c in tr[1:]:
        assert "--policy.path=" in c and "--warm.optimizer=" in c
    assert "stage0" in tr[3].split("--policy.path=")[1]   # stage 1 starts from stage 0's final policy
    merges = [c for c in cmds if "merge_data.py" in c]
    last = merges[-1].split("--inputs ")[1].split(" --out")[0].split()
    assert last[0].endswith("t3_demos") and last[1].endswith("t5_demos")   # demos first
    assert sum("rollouts_r" in x for x in last) == 4      # all rollouts of both stages kept
    assert all("--fdpg.actor_max_index=2000" in c for c in tr[3:])


def test_cold_restarts_each_stage_but_is_warm_within(tmp_path):
    tr = trains(dry("fdpg_cold", tmp_path))
    assert "--policy.type=multi_task_dit" in tr[0] and "--policy.type=multi_task_dit" in tr[3]
    assert all("--warm.optimizer=" in c for c in tr[1:3] + tr[4:])


def test_current_task_only_drops_earlier_data(tmp_path):
    cmds = dry("fdpg_current", tmp_path)
    stage1_merges = [c for c in cmds if "merge_data.py" in c and "stage1" in c.split("--out")[1]]
    for c in stage1_merges:
        assert "t3_demos" not in c and "stage0" not in c
    tr = trains(cmds)
    assert "stage0" in tr[3].split("--policy.path=")[1]   # but the policy itself carries over


def test_bc_uses_bc_mode_and_fixed_stops_training(tmp_path):
    assert all("--fdpg.mode=bc_critic" in c for c in trains(dry("bc_cumulative", tmp_path)))
    cmds = dry("fixed", tmp_path)
    assert len(trains(cmds)) == 3                         # stage 0 only
    assert not any("collect_rollouts.py" in c and "--task_id=5" in c for c in cmds)


def test_collection_seeds_are_disjoint_from_eval(tmp_path):
    cmds = dry("fdpg_cumulative", tmp_path)
    seeds = [int(c.split("--seed=")[1].split()[0]) for c in cmds if "collect_rollouts.py" in c]
    assert len(set(seeds)) == len(seeds) == 4 and min(seeds) >= 1_000_000
    assert all("--seed=0" in c for c in cmds if " eval " in c)


def test_target_needs_round_evals(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        dry("fdpg_cumulative", tmp_path, "--target", "0.9", "--round_eval_episodes", "0")


# ---------------------------------------------------------------- GPU windows and stage-0 reuse
import json  # noqa: E402
from datetime import datetime  # noqa: E402

import continual  # noqa: E402


def test_windows_block_jobs_that_would_overlap():
    w = continual.parse_windows("Mon,Thu,Fri 14:00-17:00")
    thu = datetime(2026, 11, 26, 12, 0)                        # a Thursday
    assert continual.blocked_until(thu, 3600, w) is None        # ends 13:00
    assert continual.blocked_until(thu, 3 * 3600, w) == datetime(2026, 11, 26, 17, 0)
    assert continual.blocked_until(datetime(2026, 11, 26, 15, 0), 60, w) == datetime(2026, 11, 26, 17, 0)
    assert continual.blocked_until(datetime(2026, 11, 25, 12, 0), 3 * 3600, w) is None   # Wednesday
    # a long job on Wednesday night runs into Thursday afternoon
    assert continual.blocked_until(datetime(2026, 11, 25, 23, 0), 16 * 3600, w) == datetime(2026, 11, 26, 17, 0)
    assert continual.parse_windows("none") == []


def fake_run(root: Path, order, **over):
    root.mkdir(parents=True)
    cfg = dict(suite="libero_spatial", seed=0, offline_steps=10000, rounds=4, online_steps=5000,
               collect_episodes=50, round_eval_episodes=10, target=None, eval_episodes=10, batch_size=32,
               timestep="uniform", fdpg=[], fdpg_mode="flowdpg", order=order)
    cfg.update(over)
    (root / "config.json").write_text(json.dumps(cfg))
    final = root / "stage0" / "train_r4" / "checkpoints" / "005000" / "pretrained_model"
    (root / "stages.jsonl").write_text(json.dumps({"stage": 0, "task_id": order[0], "final_ckpt": str(final),
                                                   "rollouts": [], "gradient_steps_total": 30000}) + "\n")
    (root / "online.jsonl").write_text(json.dumps({"stage": 0, "round": 0, "ckpt": str(final)}) + "\n")
    (root / "cells.jsonl").write_text(json.dumps({"stage": 0, "task": 0, "kind": "post", "n_rollouts": 10,
                                                  "successes": 9}) + "\n")
    (root / "evals.jsonl").write_text(json.dumps({"ckpt": str(final), "task_id": order[0], "n": 10, "k": 9}) + "\n")


def reuse(tmp_path, src_over=None, method="fixed", order=("3",)):
    data = tmp_path / "d"
    meta = data / "libero_spatial_tasks3" / "all_demos" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "stats.json").write_text("{}")
    fake_run(tmp_path / "src", [3], **(src_over or {}))
    return subprocess.run([sys.executable, str(SCRIPTS / "continual.py"), "--method", method, "--order", *order,
                           "--data_root", str(data), "--out", str(tmp_path / "runs"), "--avoid", "none",
                           "--stage0_from", str(tmp_path / "src")], capture_output=True, text=True)


def test_stage0_reuse_copies_records(tmp_path):
    r = reuse(tmp_path)
    assert r.returncode == 0, r.stderr
    run = tmp_path / "runs" / "fixed_libero_spatial_o3_s0"
    cells = [json.loads(x) for x in open(run / "cells.jsonl")]
    assert cells == [{**cells[0], "stage": 0, "task": 0, "kind": "post", "successes": 9}]
    assert cells[0]["method"] == "fixed" and cells[0]["run_id"] == run.name
    st = json.loads(open(run / "stages.jsonl").readline())
    assert st["reused_from"] == str(tmp_path / "src") and st["method"] == "fixed"
    assert ">> " not in r.stdout                                # nothing was trained or evaluated


def test_stage0_reuse_refuses_different_settings(tmp_path):
    r = reuse(tmp_path, {"offline_steps": 20000})
    assert r.returncode != 0 and "offline_steps" in (r.stdout + r.stderr)


# ---------------------------------------------------------------- re-evaluation with more episodes
def fake_finished(root: Path, method: str, finals: list[str]):
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"method": method, "seed": 0, "order": [0, 1, 2],
                                                  "suite": "libero_spatial"}))
    (root / "stages.jsonl").write_text("".join(json.dumps({"stage": m, "final_ckpt": f}) + "\n"
                                               for m, f in enumerate(finals)))


def test_reeval_plans_every_cell_once_per_checkpoint(tmp_path):
    main = tmp_path / "fdpg_cumulative"
    fixed = tmp_path / "fixed"
    fake_finished(main, "fdpg_cumulative", ["A", "B", "C"])
    fake_finished(fixed, "fixed", ["A", "A", "A"])          # fixed never moves from stage 0
    out = subprocess.run([sys.executable, str(SCRIPTS / "reeval.py"), "--dry_run", "--avoid", "none",
                          str(main), str(fixed)], capture_output=True, text=True, check=True).stdout
    evals = [line for line in out.splitlines() if line.startswith(">> ")]
    pairs = [(c.split("--policy.path=")[1].split()[0], c.split("--env.task_ids=[")[1].split("]")[0])
             for c in evals]
    # main: post A0, entry A1, post B0 B1, entry B2, post C0 C1 C2 = 8; fixed adds only A on task 2
    assert len(pairs) == len(set(pairs)) == 9
    assert ("A", "2") in pairs and all("--eval.n_episodes=50" in c and "--seed=0" in c for c in evals)
    cells = [json.loads(line[5:]) for line in out.splitlines() if line.startswith("CELL ")]
    assert len(cells) == 16                                  # 8 cells per 3-task run
    assert {c["kind"] for c in cells if c["stage"] == 0} == {"post"}


def test_reeval_refuses_unfinished_runs(tmp_path):
    run = tmp_path / "r"
    fake_finished(run, "fdpg_current", ["A"])
    r = subprocess.run([sys.executable, str(SCRIPTS / "reeval.py"), "--dry_run", str(run)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "not finished" in (r.stdout + r.stderr)
