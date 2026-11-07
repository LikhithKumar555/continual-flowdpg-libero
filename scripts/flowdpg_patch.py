"""FlowDPG (offline) on top of LeRobot's MultiTaskDiT flow-matching policy.

Applied as a runtime patch, so LeRobot's own trainer, data loading, normalisation,
checkpointing and lerobot-eval are reused unchanged. Use it through flowdpg_launch.py.

What the patch changes
----------------------
1. Data: observation_delta_indices gets K = n_action_steps extra future steps, so every
   sample also carries the observation window at t+K (the state after the executed chunk).
   The dataset's "observation.state_is_pad" at t+K tells us the episode ended inside the
   executed chunk -> terminal transition with reward 1 (all demos are successes).
2. Critic (IQL, Kostrikov et al.): twin Q(s, a_exec) and V(s) as small MLPs on the policy's
   own conditioning vector, DETACHED (the critic does not train the vision/text encoders).
     V  : expectile regression on min target-Q of the data action, expectile tau.
     Q  : MSE to r + gamma^K * (1 - done) * V(s').
   a_exec = the K executed actions of the chunk (what the robot actually does).
3. Actor (FlowDPG): for the sampled flow time t,
     a_hat = x_t + (1 - t) v(x_t, t, s)                       one-step action estimate
     g     = d min(Q1, Q2)(s, a_hat_exec) / d a_hat           (zero outside the executed part)
     a*    = a_hat + alpha * ||u_t|| / (||g|| + eps) * g        critic-improved target
     u*    = a* - (1 - sigma_min) x_0
     L     = lam * ||v - u*||^2 + (1 - lam) * ||v - u_t||^2  + w_c * ||a_hat - a_data||^2
   lam ramps linearly 0 -> lam_max over lam_warmup steps after lam_start (paper: 2000, 0.5).
   No backprop through the ODE; inference stays the normal Euler solve.
4. Optimiser: critic parameters get their own group with lr = critic_lr.
5. update(): Polyak update of the target Q networks and the step counter.

Everything else (sampling, inference, saving) is the original MultiTaskDiT code.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"
REWARD = "next.reward"


@dataclass
class FDPGConfig:
    mode: str = "flowdpg"          # "flowdpg" or "bc_critic" (critic trained, actor = plain BC: diagnostic)
    alpha: float = 0.5             # step size of the critic push, relative to ||u_t||
    lam_max: float = 0.5           # weight of the critic-improved target
    lam_warmup: int = 2000         # steps to ramp lam from 0 to lam_max
    lam_start: int = 0             # step at which the ramp starts (lets the critic warm up first)
    consistency_weight: float = 1.0
    expectile: float = 0.7
    gamma: float = 0.99            # per environment step; chunk discount is gamma ** K
    terminal_reward: float = 1.0
    critic_lr: float = 3e-4
    target_tau: float = 0.005
    hidden: int = 512
    critic_arch: str = "split"     # "split": separate obs / action projections (default); "concat": one MLP on [obs, act]
    proj_dim: int = 256            # width of each projection in the split critic
    critic_weight: float = 1.0
    actor_data: str = "success"    # actor learns from "success" episodes only (critic: all data), or "all"
    clamp_actions: bool = True     # clamp normalised action targets to [-1, 1] (= demo range with demo stats)
    eps: float = 1e-6

    @classmethod
    def from_args(cls, items: dict[str, str]) -> "FDPGConfig":
        cfg = cls()
        for k, v in items.items():
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown --fdpg.{k}; valid: {list(asdict(cfg))}")
            typ = type(getattr(cfg, k))
            if typ is bool:
                if str(v).lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise SystemExit(f"--fdpg.{k} must be true or false, got {v!r}")
                setattr(cfg, k, str(v).lower() in ("true", "1", "yes"))
            else:
                setattr(cfg, k, typ(v))
        if cfg.mode not in ("flowdpg", "bc_critic"):
            raise SystemExit("--fdpg.mode must be flowdpg or bc_critic")
        if cfg.critic_arch not in ("split", "concat"):
            raise SystemExit("--fdpg.critic_arch must be split or concat")
        if cfg.actor_data not in ("success", "all"):
            raise SystemExit("--fdpg.actor_data must be success or all")
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, 1),
    )


class SplitQHead(nn.Module):
    """Q(s, a) with separate projections, so the ~4k-dim observation features cannot drown out
    the action (168 numbers): obs -> proj_dim, act -> proj_dim, then an MLP on the concatenation."""

    def __init__(self, cond_dim: int, act_dim: int, hidden: int, proj_dim: int):
        super().__init__()
        self.obs = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, proj_dim),
                                 nn.LayerNorm(proj_dim), nn.Mish())
        self.act = nn.Sequential(nn.Linear(act_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Mish())
        self.mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.obs(cond), self.act(act)], dim=-1))


class TwinQ(nn.Module):
    def __init__(self, cond_dim: int, act_dim: int, hidden: int, arch: str = "split", proj_dim: int = 256):
        super().__init__()
        self.arch = arch
        if arch == "split":
            self.q1 = SplitQHead(cond_dim, act_dim, hidden, proj_dim)
            self.q2 = SplitQHead(cond_dim, act_dim, hidden, proj_dim)
        elif arch == "concat":
            # parameter names stay q1.<i>/q2.<i> as in the original checkpoints
            self.q1 = _mlp(cond_dim + act_dim, hidden)
            self.q2 = _mlp(cond_dim + act_dim, hidden)
        else:
            raise ValueError(f"unknown critic arch {arch!r}")

    def forward(self, cond: torch.Tensor, act: torch.Tensor):
        if self.arch == "concat":
            x = torch.cat([cond, act], dim=-1)
            return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
        return self.q1(cond, act).squeeze(-1), self.q2(cond, act).squeeze(-1)


def critic_settings_from_checkpoint(pretrained_dir) -> dict:
    """Read the critic architecture of a saved checkpoint from its weight names / shapes.

    Returns {} if the checkpoint has no FlowDPG critic (plain BC), else
    {"critic_arch": ..., "hidden": ..., ("proj_dim": ...)} to pass into FDPGConfig.
    """
    from pathlib import Path

    from safetensors import safe_open

    path = Path(pretrained_dir) / "model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        if not any(k.startswith("fdpg_q.") for k in keys):
            return {}
        if "fdpg_q.q1.obs.1.weight" in keys:
            proj, _ = f.get_slice("fdpg_q.q1.obs.1.weight").get_shape()
            hidden, _ = f.get_slice("fdpg_q.q1.mlp.0.weight").get_shape()
            return {"critic_arch": "split", "hidden": int(hidden), "proj_dim": int(proj)}
        hidden, _ = f.get_slice("fdpg_q.q1.1.weight").get_shape()
        return {"critic_arch": "concat", "hidden": int(hidden)}


class ValueNet(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.v = _mlp(cond_dim, hidden)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.v(cond).squeeze(-1)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """IQL: |tau - 1(diff < 0)| * diff^2, diff = Q - V."""
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


def lambda_at(step: int, hp: FDPGConfig) -> float:
    if hp.mode != "flowdpg" or hp.lam_max <= 0:
        return 0.0
    if hp.lam_warmup <= 0:
        return hp.lam_max if step >= hp.lam_start else 0.0
    return hp.lam_max * min(1.0, max(0.0, (step - hp.lam_start) / hp.lam_warmup))


def critic_push(q_net: TwinQ, cond: torch.Tensor, a_hat: torch.Tensor, u_t: torch.Tensor,
                s: int, k: int, alpha: float, eps: float):
    """Return (delta, grad-norm) with delta = alpha * ||u_t|| / (||g|| + eps) * g, g = dminQ/da_hat.

    a_hat, u_t: (B, H, A). Only the executed slice [s, s+k) is scored by the critic.
    Gradients are taken w.r.t. the action only; critic parameters receive no .grad here.
    """
    b = a_hat.shape[0]
    a_exec = a_hat[:, s:s + k].detach().reshape(b, -1).requires_grad_(True)
    with torch.enable_grad():
        q1, q2 = q_net(cond.detach(), a_exec)
        g = torch.autograd.grad(torch.minimum(q1, q2).sum(), a_exec)[0]
    g_full = torch.zeros_like(a_hat)
    g_full[:, s:s + k] = g.view(b, k, -1)
    u_norm = u_t.reshape(b, -1).norm(dim=1).view(b, 1, 1)
    g_norm = g_full.reshape(b, -1).norm(dim=1).view(b, 1, 1)
    delta = alpha * u_norm / (g_norm + eps) * g_full
    return delta.detach(), g_norm.mean().detach()


def chunk_reward_and_done(batch: dict, bsz: int, k: int, hp: "FDPGConfig"):
    """Reward and terminal flag of the chunk transition (s_t, a_t..a_{t+K-1}) -> s_{t+K}.

    If the dataset has a "next.reward" column (rollouts, or demos prepared by
    build_task_dataset.py) the patch requests its K-step window: reward = sum_j gamma^j r_{t+j},
    terminal = any reward > 0 (LIBERO ends an episode on success; reward is 1 only then).
    An episode that ends without success (time limit) is NOT terminal: V(s') is bootstrapped from
    the last frame (LeRobot pads past the end by repeating it).
    Without a reward column (raw lerobot/libero demos, all successful) the end of the demo inside
    the chunk counts as success, read from "observation.state_is_pad" at t+K.
    Returns (reward, done, "data" | "inferred").
    """
    rew = batch.get(REWARD)
    if rew is not None:
        rew = rew.reshape(bsz, -1).float()
        if rew.shape[1] != k:
            raise RuntimeError(f"{REWARD} window has {rew.shape[1]} steps, expected {k}")
        rpad = batch.get(f"{REWARD}_is_pad")
        if rpad is not None:
            rew = rew.masked_fill(rpad.reshape(bsz, -1).bool(), 0.0)
        w = hp.gamma ** torch.arange(k, device=rew.device, dtype=rew.dtype)
        return (rew * w).sum(1), (rew > 0).any(1).float(), "data"
    pad = batch.get(f"{OBS_STATE}_is_pad")
    if pad is None:
        raise RuntimeError(f"neither {REWARD} nor {OBS_STATE}_is_pad in batch: cannot tell episode ends")
    done = pad[:, -1].float()
    return done * hp.terminal_reward, done, "inferred"


EP_SUCCESS = "episode_success"


def actor_weights(batch: dict, bsz: int, hp: "FDPGConfig", device) -> torch.Tensor:
    """Per-sample actor weights (B,). 1 everywhere unless actor_data="success" and the dataset
    has an episode_success column, then 1 for frames of successful episodes and 0 otherwise."""
    ones = torch.ones(bsz, device=device)
    if hp.actor_data == "all":
        return ones
    if EP_SUCCESS not in batch:
        if REWARD in batch:
            # Our datasets (demos built by build_task_dataset.py, rollouts) carry both columns, so a
            # missing flag means it was dropped on the way: fail instead of silently imitating failures.
            raise RuntimeError(
                f"actor_data=success but '{EP_SUCCESS}' is missing from the batch while '{REWARD}' is present. "
                "Was the dataset built with the current libero_data.py, and is the patch's processor "
                "key registration active (keep_episode_success_key)?")
        return ones                                   # raw lerobot/libero demos: all successful
    return batch[EP_SUCCESS].reshape(bsz, -1)[:, 0].to(device=device, dtype=torch.float32)


def keep_episode_success_key() -> bool:
    """LeRobot's batch->transition converter keeps only a fixed list of extra keys (plus *_is_pad) and
    drops everything else before the policy sees the batch. Add episode_success to that list.
    Returns True if registered (or already present)."""
    try:
        from lerobot.processor import converters
    except ImportError:
        return False
    keys = getattr(converters, "_COMPLEMENTARY_KEYS", None)
    if keys is None:
        return False
    if EP_SUCCESS not in keys:
        converters._COMPLEMENTARY_KEYS = tuple(keys) + (EP_SUCCESS,)
    return True


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample, averaged with weights w (B,). All-zero weights give 0."""
    per = (pred - target).pow(2).reshape(pred.shape[0], -1).mean(dim=1)
    return (per * w).sum() / w.sum().clamp_min(1.0)


def split_time(batch: dict, n_obs: int):
    """Split the stacked (current + next) observation windows into two batches."""
    cur, nxt = dict(batch), dict(batch)
    for key in (OBS_STATE, OBS_IMAGES):
        if key in batch:
            x = batch[key]
            if x.shape[1] != 2 * n_obs:
                raise RuntimeError(f"{key} has {x.shape[1]} time steps, expected {2 * n_obs} "
                                   "(is the FlowDPG data patch active?)")
            cur[key], nxt[key] = x[:, :n_obs], x[:, n_obs:]
    return cur, nxt


# ---------------------------------------------------------------- patched methods

def _patched_obs_delta_indices(self) -> list:
    k = self.n_action_steps
    cur = list(range(1 - self.n_obs_steps, 1))
    return cur + [i + k for i in cur]


def _patched_reward_delta_indices(self) -> list:
    """Rewards of the K executed actions a_t .. a_{t+K-1} (used only if the dataset has next.reward)."""
    return list(range(0, self.n_action_steps))


def _make_init(orig_init, hp: FDPGConfig):
    def __init__(self, config, **kwargs):
        orig_init(self, config, **kwargs)
        cond_dim = self.observation_encoder.conditioning_dim
        act_dim = config.action_feature.shape[0] * config.n_action_steps
        self.fdpg_q = TwinQ(cond_dim, act_dim, hp.hidden, arch=hp.critic_arch, proj_dim=hp.proj_dim)
        self.fdpg_q_target = copy.deepcopy(self.fdpg_q)
        for p in self.fdpg_q_target.parameters():
            p.requires_grad_(False)
        self.fdpg_v = ValueNet(cond_dim, hp.hidden)
        self.register_buffer("fdpg_step", torch.zeros((), dtype=torch.long))
        self._fdpg_hp = hp
    return __init__


def _make_get_optim_params(orig):
    def get_optim_params(self):
        critic = [p for n, p in self.named_parameters()
                  if p.requires_grad and (n.startswith("fdpg_q.") or n.startswith("fdpg_v."))]
        ids = {id(p) for p in critic}
        groups = []
        for g in orig(self):
            params = [p for p in g["params"] if id(p) not in ids]
            if params:
                groups.append({**g, "params": params})
        groups.append({"params": critic, "lr": self._fdpg_hp.critic_lr})
        return groups
    return get_optim_params


def _update(self):
    tau = self._fdpg_hp.target_tau
    with torch.no_grad():
        for p, tp in zip(self.fdpg_q.parameters(), self.fdpg_q_target.parameters()):
            tp.mul_(1.0 - tau).add_(p, alpha=tau)
    self.fdpg_step += 1


def _forward(self, batch: dict, reduction: str = "mean"):
    if reduction != "mean":
        raise NotImplementedError("FlowDPG patch supports reduction='mean' only")
    hp: FDPGConfig = self._fdpg_hp
    cfg = self.config
    n_obs, k = cfg.n_obs_steps, cfg.n_action_steps
    s = n_obs - 1                                    # first executed action in the chunk

    batch = self._prepare_batch(batch)
    cur, nxt = split_time(batch, n_obs)
    cond = self.observation_encoder.encode(cur)
    with torch.no_grad():
        cond_next = self.observation_encoder.encode(nxt)
    c = cond.detach()

    data = batch[ACTION]                             # (B, H, A), normalised
    if hp.clamp_actions:
        # MIN_MAX-normalised with the demo stats (kept fixed across online rounds), so [-1, 1] is the
        # demo range: never regress towards (or score) actions outside what the demos did.
        data = data.clamp(-1.0, 1.0)
    bsz = data.shape[0]
    a_exec = data[:, s:s + k].reshape(bsz, -1)

    reward, done, source = chunk_reward_and_done(batch, bsz, k, hp)
    disc = hp.gamma ** k

    # ---- critic (IQL)
    with torch.no_grad():
        tq1, tq2 = self.fdpg_q_target(c, a_exec)
        q_tgt = torch.minimum(tq1, tq2)
        y = reward + disc * (1.0 - done) * self.fdpg_v(cond_next)
    v = self.fdpg_v(c)
    v_loss = expectile_loss(q_tgt - v, hp.expectile)
    q1, q2 = self.fdpg_q(c, a_exec)
    q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    critic_loss = v_loss + q_loss

    # ---- actor (FlowDPG on the flow-matching objective)
    obj = self.objective
    sigma = cfg.sigma_min
    noise = torch.randn_like(data)
    t = obj._sample_timesteps(bsz, data.device)
    te = t.view(-1, 1, 1)
    x_t = te * data + (1 - (1 - sigma) * te) * noise
    u_t = data - (1 - sigma) * noise
    v_pred = self.noise_predictor(x_t, t, conditioning_vec=cond)
    a_hat = x_t + (1 - te) * v_pred

    # Which samples the actor learns from: with actor_data="success" only frames of episodes that
    # ended in success (demos, successful rollouts). Every actor term regresses towards the data
    # action, so failed rollouts would otherwise be imitated. The critic above uses all data.
    w = actor_weights(batch, bsz, hp, data.device)

    bc_loss = weighted_mse(v_pred, u_t, w)
    lam = lambda_at(int(self.fdpg_step.item()), hp)
    g_norm = torch.zeros((), device=data.device)
    if lam > 0:
        delta, g_norm = critic_push(self.fdpg_q, c, a_hat, u_t, s, k, hp.alpha, hp.eps)
        a_star = a_hat.detach() + delta
        u_star = a_star - (1 - sigma) * noise
        rl_loss = weighted_mse(v_pred, u_star, w)
        actor_loss = lam * rl_loss + (1 - lam) * bc_loss
    else:
        rl_loss = torch.zeros((), device=data.device)
        actor_loss = bc_loss
    cons_loss = weighted_mse(a_hat, data, w)
    actor_loss = actor_loss + hp.consistency_weight * cons_loss

    loss = actor_loss + hp.critic_weight * critic_loss
    info = {
        "fdpg_bc": bc_loss.item(), "fdpg_rl": rl_loss.item(), "fdpg_cons": cons_loss.item(),
        "fdpg_lambda": lam, "fdpg_v_loss": v_loss.item(), "fdpg_q_loss": q_loss.item(),
        "fdpg_q_mean": q1.mean().item(), "fdpg_v_mean": v.mean().item(),
        "fdpg_done_frac": done.mean().item(), "fdpg_grad_norm": float(g_norm),
        "fdpg_reward_from_data": float(source == "data"),
        "fdpg_actor_frac": float((w > 0).float().mean()),
    }
    return loss, info


def install(hp: FDPGConfig, policy_cls=None, config_cls=None) -> None:
    """Patch MultiTaskDiT in place. Call before LeRobot builds the dataset or the policy."""
    if policy_cls is None or config_cls is None:
        from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
        from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
        policy_cls = policy_cls or MultiTaskDiTPolicy
        config_cls = config_cls or MultiTaskDiTConfig
    if getattr(policy_cls, "_fdpg_installed", False):
        return
    if not keep_episode_success_key() and hp.actor_data == "success":
        print("[flowdpg] WARNING: could not register episode_success with LeRobot's processor; "
              "training on rollout data will stop with an error", flush=True)
    config_cls.observation_delta_indices = property(_patched_obs_delta_indices)
    config_cls.reward_delta_indices = property(_patched_reward_delta_indices)
    policy_cls.__init__ = _make_init(policy_cls.__init__, hp)
    policy_cls.get_optim_params = _make_get_optim_params(policy_cls.get_optim_params)
    policy_cls.forward = _forward
    policy_cls.update = _update
    policy_cls._fdpg_installed = True
