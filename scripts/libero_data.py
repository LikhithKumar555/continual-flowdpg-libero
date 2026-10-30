"""Dataset helpers shared by build_task_dataset.py, collect_rollouts.py and merge_data.py.

One schema for everything the agent trains on, demos and rollouts alike:
  observation.images.* (the demo cameras), observation.state (8: eef pos, axis-angle, gripper),
  action (7), next.reward (1, float), next.success (1, bool), next.done (1, bool), task.
Same fps and robot_type as lerobot/libero, so LeRobot's merge_datasets accepts them together.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

OBS_STATE = "observation.state"
ACTION = "action"
REWARD = "next.reward"
SUCCESS = "next.success"
DONE = "next.done"
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
SRC_REPO = "lerobot/libero"

LABEL_SPECS = {
    REWARD: {"dtype": "float32", "shape": [1], "names": None},
    SUCCESS: {"dtype": "bool", "shape": [1], "names": None},
    DONE: {"dtype": "bool", "shape": [1], "names": None},
}


def work_dir() -> Path:
    return Path(os.environ.get("WORK", "."))


def kept_keys(features: dict) -> list[str]:
    """Feature keys both demos and rollouts provide: cameras, 8-d state, action."""
    cams = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]
    missing = [k for k in (OBS_STATE, ACTION) if k not in features]
    if missing or not cams:
        raise RuntimeError(f"unexpected source features, missing {missing}, cameras {cams}")
    return cams + [OBS_STATE, ACTION]


def rollout_features(src_features: dict) -> dict:
    """Schema for recorded rollouts: the kept demo features (encoder info stripped) + labels."""
    out = {}
    for k in kept_keys(src_features):
        spec = {kk: vv for kk, vv in src_features[k].items() if kk != "info"}
        out[k] = spec
    out.update({k: dict(v) for k, v in LABEL_SPECS.items()})
    return out


def episodes_for_env_task(suite: str, task_id: int, task_episodes: Path | None = None) -> tuple[str, list[int]]:
    """(instruction text, dataset episode indices) for one LIBERO env task, matched by text."""
    from libero.libero import benchmark

    text = benchmark.get_benchmark_dict()[suite]().get_task(task_id).language
    d = json.load(open(task_episodes or work_dir() / "task_episodes.json"))
    hits = [v for v in d.values() if v["text"].strip().lower() == text.strip().lower()]
    if len(hits) != 1:
        raise SystemExit(f"no unique dataset task for {suite} task {task_id} ({text!r}): {len(hits)} matches")
    return text, hits[0]["episodes"]


def last_frame_labels(episode_index: np.ndarray, frame_index: np.ndarray):
    """Per-frame (is_last) for demo episodes: success / reward / done only on the final frame."""
    episode_index = np.asarray(episode_index)
    frame_index = np.asarray(frame_index)
    last = {}
    for e, f in zip(episode_index, frame_index):
        last[int(e)] = max(last.get(int(e), -1), int(f))
    return np.array([int(f) == last[int(e)] for e, f in zip(episode_index, frame_index)])


def build_demo_dataset(src, episodes: list[int], out_dir: Path, repo_id: str):
    """Copy `episodes` of the source dataset into out_dir with the shared schema.

    src: a LeRobotDataset (normally lerobot/libero). Extra source features are dropped,
    next.reward / next.success / next.done are set on the last frame of every demo.
    """
    from lerobot.datasets.dataset_tools import modify_features, split_dataset

    out_dir = Path(out_dir)
    tmp = out_dir.parent / f"_tmp_split_{out_dir.name}"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; delete it or choose another --out")
    if tmp.exists():
        shutil.rmtree(tmp)
    sub = split_dataset(src, {"demos": sorted(episodes)}, output_dir=tmp)["demos"]

    ep_len: dict[int, int] = {}
    for e in np.asarray(sub.hf_dataset["episode_index"]):
        ep_len[int(e)] = ep_len.get(int(e), 0) + 1

    def is_last(_row, ep, fr):
        return int(fr) == ep_len[int(ep)] - 1

    keep = set(kept_keys(sub.meta.features)) | DEFAULT_FEATURES
    remove = [k for k in sub.meta.features if k not in keep]
    out = modify_features(
        sub,
        add_features={
            REWARD: (lambda r, e, f: np.float32(1.0 if is_last(r, e, f) else 0.0), LABEL_SPECS[REWARD]),
            SUCCESS: (lambda r, e, f: bool(is_last(r, e, f)), LABEL_SPECS[SUCCESS]),
            DONE: (lambda r, e, f: bool(is_last(r, e, f)), LABEL_SPECS[DONE]),
        },
        remove_features=remove or None,
        output_dir=out_dir,
        repo_id=repo_id,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def image_to_uint8_hwc(img) -> np.ndarray:
    """(C,H,W) float in [0,1] (torch or numpy) -> (H,W,C) uint8."""
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def episode_frames(obs_steps: list[dict], actions: list, env_idx: int, successes, done_idx: int,
                   task: str, cam_keys: list[str]) -> list[dict]:
    """Dataset frames t = 0..done_idx for one env of a batched rollout.

    obs_steps[t]: processed observation dict (batch tensors) seen by the policy at step t.
    actions[t]:   (B, A) action in dataset units at step t.
    successes:    (B, T) success flags from the rollout. Reward = 1 on the success step, else 0
                  (sparse success reward, as agreed with the supervisor).
    """
    frames = []
    for t in range(done_idx + 1):
        o = obs_steps[t]
        frame = {k: image_to_uint8_hwc(o[k][env_idx]) for k in cam_keys}
        st = o[OBS_STATE][env_idx]
        frame[OBS_STATE] = (st.detach().cpu().numpy() if hasattr(st, "detach") else np.asarray(st)).astype(np.float32)
        a = actions[t][env_idx]
        frame[ACTION] = (a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)).astype(np.float32)
        ok = bool(successes[env_idx][t])
        frame[REWARD] = np.array([1.0 if ok else 0.0], dtype=np.float32)
        frame[SUCCESS] = np.array([ok])
        frame[DONE] = np.array([t == done_idx])
        frame["task"] = task
        frames.append(frame)
    return frames


def first_done(done_row) -> int:
    """Index of the first True in a cumulative done row (last index if never done)."""
    arr = np.asarray(done_row).astype(bool)
    idx = np.flatnonzero(arr)
    return int(idx[0]) if idx.size else len(arr) - 1
