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
    critic_weight: float = 1.0
    eps: float = 1e-6

    @classmethod
    def from_args(cls, items: dict[str, str]) -> "FDPGConfig":
        cfg = cls()
        for k, v in items.items():
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown --fdpg.{k}; valid: {list(asdict(cfg))}")
            typ = type(getattr(cfg, k))
            setattr(cfg, k, typ(v))
        if cfg.mode not in ("flowdpg", "bc_critic"):
            raise SystemExit("--fdpg.mode must be flowdpg or bc_critic")
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


class TwinQ(nn.Module):
    def __init__(self, cond_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = _mlp(cond_dim + act_dim, hidden)
        self.q2 = _mlp(cond_dim + act_dim, hidden)

    def forward(self, cond: torch.Tensor, act: torch.Tensor):
        x = torch.cat([cond, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


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


def _make_init(orig_init, hp: FDPGConfig):
    def __init__(self, config, **kwargs):
        orig_init(self, config, **kwargs)
        cond_dim = self.observation_encoder.conditioning_dim
        act_dim = config.action_feature.shape[0] * config.n_action_steps
        self.fdpg_q = TwinQ(cond_dim, act_dim, hp.hidden)
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
    bsz = data.shape[0]
    a_exec = data[:, s:s + k].reshape(bsz, -1)

    pad = batch.get(f"{OBS_STATE}_is_pad")
    if pad is None:
        raise RuntimeError(f"{OBS_STATE}_is_pad missing from batch: cannot tell episode ends")
    done = pad[:, -1].float()                        # obs at t+K lies past the episode end
    reward = done * hp.terminal_reward
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

    bc_loss = F.mse_loss(v_pred, u_t)
    lam = lambda_at(int(self.fdpg_step.item()), hp)
    g_norm = torch.zeros((), device=data.device)
    if lam > 0:
        delta, g_norm = critic_push(self.fdpg_q, c, a_hat, u_t, s, k, hp.alpha, hp.eps)
        a_star = a_hat.detach() + delta
        u_star = a_star - (1 - sigma) * noise
        rl_loss = F.mse_loss(v_pred, u_star)
        actor_loss = lam * rl_loss + (1 - lam) * bc_loss
    else:
        rl_loss = torch.zeros((), device=data.device)
        actor_loss = bc_loss
    cons_loss = F.mse_loss(a_hat, data)
    actor_loss = actor_loss + hp.consistency_weight * cons_loss

    loss = actor_loss + hp.critic_weight * critic_loss
    info = {
        "fdpg_bc": bc_loss.item(), "fdpg_rl": rl_loss.item(), "fdpg_cons": cons_loss.item(),
        "fdpg_lambda": lam, "fdpg_v_loss": v_loss.item(), "fdpg_q_loss": q_loss.item(),
        "fdpg_q_mean": q1.mean().item(), "fdpg_v_mean": v.mean().item(),
        "fdpg_done_frac": done.mean().item(), "fdpg_grad_norm": float(g_norm),
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
    config_cls.observation_delta_indices = property(_patched_obs_delta_indices)
    policy_cls.__init__ = _make_init(policy_cls.__init__, hp)
    policy_cls.get_optim_params = _make_get_optim_params(policy_cls.get_optim_params)
    policy_cls.forward = _forward
    policy_cls.update = _update
    policy_cls._fdpg_installed = True
