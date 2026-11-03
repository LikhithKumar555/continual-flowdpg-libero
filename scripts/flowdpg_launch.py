#!/usr/bin/env python
"""Run lerobot-train / lerobot-eval with the FlowDPG patch applied.

  python flowdpg_launch.py train [--fdpg.KEY=VALUE ...] <normal lerobot-train arguments>
  python flowdpg_launch.py eval  <normal lerobot-eval arguments>

--fdpg.* flags are consumed here (see FDPGConfig in flowdpg_patch.py); everything else goes
to LeRobot unchanged. Checkpoints contain the critic too, so evaluate them through this
launcher ("eval"); the critic architecture is read from the checkpoint automatically.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowdpg_patch import FDPGConfig, critic_settings_from_checkpoint, install  # noqa: E402


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    fdpg, rest = {}, []
    for arg in sys.argv[2:]:
        if arg.startswith("--fdpg."):
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
