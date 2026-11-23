#!/usr/bin/env python
"""Run lerobot-train / lerobot-eval with the FlowDPG patch applied.

  python flowdpg_launch.py train [--fdpg.KEY=VALUE ...] <normal lerobot-train arguments>
  python flowdpg_launch.py eval  <normal lerobot-eval arguments>

--fdpg.* flags are consumed here (see FDPGConfig in flowdpg_patch.py); everything else goes
to LeRobot unchanged. --warm.optimizer=<checkpoint dir> (train only) also loads that checkpoint's
optimizer state (Adam moments), see warm_start.py; use it together with --policy.path.
At exit the peak GPU memory is printed as "[flowdpg] peak_gpu_mem_gb=<x>".
Checkpoints contain the critic too, so evaluate them through this
launcher ("eval"); the critic architecture is read from the checkpoint automatically.
"""
import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowdpg_patch import FDPGConfig, critic_settings_from_checkpoint, install  # noqa: E402


def _print_peak_memory():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[flowdpg] peak_gpu_mem_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    fdpg, rest, warm_opt = {}, [], None
    for arg in sys.argv[2:]:
        if arg.startswith("--warm.optimizer="):
            warm_opt = arg.split("=", 1)[1]
        elif arg.startswith("--fdpg."):
            key, _, val = arg[len("--fdpg."):].partition("=")
            fdpg[key] = val
        else:
            rest.append(arg)
    # A checkpoint passed with --policy.path fixes the critic architecture (eval or warm start):
    # read it from the weights so old (concat) and new (split) checkpoints both load.
    ckpt = next((a.split("=", 1)[1] for a in rest if a.startswith("--policy.path=")), None)
    if ckpt:
        found = critic_settings_from_checkpoint(ckpt)
        for k, v in found.items():
            if k in fdpg and str(fdpg[k]) != str(v):
                raise SystemExit(f"--fdpg.{k}={fdpg[k]} conflicts with the checkpoint ({k}={v})")
            fdpg.setdefault(k, str(v))
        if not found:
            print("[flowdpg] checkpoint has no critic: a new critic is initialised", flush=True)
    hp = FDPGConfig.from_args(fdpg)
    print("[flowdpg] config:", hp.to_json().replace("\n", ""), flush=True)
    install(hp)
    if warm_opt:
        if mode != "train":
            raise SystemExit("--warm.optimizer is for train only")
        import warm_start
        warm_start.install(warm_opt)
    atexit.register(_print_peak_memory)

    if mode == "train":
        out = next((a.split("=", 1)[1] for a in rest if a.startswith("--output_dir=")), None)
        if out:
            side = Path(out.rstrip("/") + "_fdpg.json")
            side.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(hp.to_json())
        from lerobot.scripts.lerobot_train import main as lerobot_main
        sys.argv = ["lerobot-train", *rest]
    else:
        from lerobot.scripts.lerobot_eval import main as lerobot_main
        sys.argv = ["lerobot-eval", *rest]
    os.environ.setdefault("MUJOCO_GL", "egl")
    lerobot_main()


if __name__ == "__main__":
    main()
