"""Carry the optimizer state of a checkpoint into a new lerobot-train run (warm start).

lerobot-train with --policy.path loads the weights only and builds a fresh optimizer. For the
continual method the Overview asks to warm-start policy, critic AND optimizer, so the launcher
wraps LeRobot's make_optimizer_and_scheduler and, right after the optimizer is built, copies in
the per-parameter state (Adam moments and step counts) saved in <checkpoint>/training_state.

Only the per-parameter state is copied. The param groups (learning rates, betas, weight decay)
and the learning-rate scheduler come from the new run, so each stage starts its own schedule.
The checkpoint must come from the same model (policy + critic): parameter count and every
state tensor's shape are checked, and any mismatch is an error rather than a silent fresh start.
"""
from __future__ import annotations

from pathlib import Path

OPTIMIZER_STATE = "optimizer_state.safetensors"


def training_state_dir(path: str | Path) -> Path:
    """Accept a checkpoint step dir, its pretrained_model dir or its training_state dir."""
    p = Path(path)
    for cand in (p, p / "training_state", p.parent / "training_state"):
        if (cand / OPTIMIZER_STATE).exists():
            return cand
    raise FileNotFoundError(f"no {OPTIMIZER_STATE} under {p} (or its training_state folder)")


def unflatten_state(flat: dict) -> dict[int, dict]:
    """{'state/3/exp_avg': t, ...} (LeRobot's flatten_dict, sep '/') -> {3: {'exp_avg': t}}."""
    out: dict[int, dict] = {}
    for key, val in flat.items():
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "state":
            continue
        out.setdefault(int(parts[1]), {})[parts[2]] = val
    return out


def load_param_state(optimizer, path: str | Path) -> int:
    """Copy the saved per-parameter state into `optimizer`; returns the number of parameters filled."""
    from safetensors.torch import load_file

    src = training_state_dir(path)
    saved = unflatten_state(load_file(str(src / OPTIMIZER_STATE)))
    params = [p for g in optimizer.param_groups for p in g["params"]]
    if not saved:
        raise ValueError(f"{src}: optimizer state is empty")
    if max(saved) >= len(params):
        raise ValueError(f"{src}: saved state covers parameter {max(saved)}, optimizer has {len(params)}")
    for i, st in saved.items():
        for name, t in st.items():
            if t.dim() > 0 and tuple(t.shape) != tuple(params[i].shape):
                raise ValueError(f"{src}: state {name} of parameter {i} has shape {tuple(t.shape)}, "
                                 f"parameter has {tuple(params[i].shape)} (different model?)")
    sd = optimizer.state_dict()
    sd["state"] = saved
    optimizer.load_state_dict(sd)   # keeps the new param groups; moves state to the params' device
    return len(saved)


def install(path: str | Path) -> None:
    """Make lerobot-train load the optimizer state from `path` when it builds its optimizer."""
    import lerobot.scripts.lerobot_train as lt

    original = lt.make_optimizer_and_scheduler
    src = training_state_dir(path)

    def wrapped(*args, **kwargs):
        optimizer, scheduler = original(*args, **kwargs)
        opts = optimizer.values() if isinstance(optimizer, dict) else [optimizer]
        for opt in opts:
            n = load_param_state(opt, src)
            print(f"[warm_start] optimizer state for {n} parameters loaded from {src}", flush=True)
        return optimizer, scheduler

    lt.make_optimizer_and_scheduler = wrapped
