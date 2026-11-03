"""CPU tests of the FlowDPG patch against a stand-in with MultiTaskDiT's interface.

The stand-in mirrors what the patch touches: config.{n_obs_steps, n_action_steps, horizon,
sigma_min, action_feature}, observation_encoder.{encode, conditioning_dim}, noise_predictor,
objective._sample_timesteps, _prepare_batch, get_optim_params. The real CLIP model is not used.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flowdpg_patch as fp  # noqa: E402

A, H, NOBS, K, SD = 3, 8, 2, 4, 5


@dataclass
class _Feat:
    shape: tuple


@dataclass
class FakeConfig:
    n_obs_steps: int = NOBS
    horizon: int = H
    n_action_steps: int = K
    sigma_min: float = 0.0
    action_feature: _Feat = field(default_factory=lambda: _Feat((A,)))

    @property
    def observation_delta_indices(self):
        return list(range(1 - self.n_obs_steps, 1))


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(SD + 1, 6)
        self.conditioning_dim = 6 * NOBS

    def encode(self, batch):
        st = batch[fp.OBS_STATE]                               # (B, T, SD)
        img = batch[fp.OBS_IMAGES].mean(dim=(2, 3, 4, 5)).unsqueeze(-1)  # (B, T, 1)
        return self.lin(torch.cat([st, img], -1)).flatten(1)


class FakeVel(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H * A + 1 + cond_dim, 64), nn.Mish(), nn.Linear(64, H * A))

    def forward(self, x, t, conditioning_vec):
        b = x.shape[0]
        return self.net(torch.cat([x.flatten(1), t.view(-1, 1), conditioning_vec], -1)).view(b, H, A)


class FakeObjective:
    def _sample_timesteps(self, b, device):
        return torch.rand(b, device=device)


class FakePolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_encoder = FakeEncoder()
        self.noise_predictor = FakeVel(self.observation_encoder.conditioning_dim)
        self.objective = FakeObjective()

    def _prepare_batch(self, batch):
        return batch

    def get_optim_params(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]


HP = fp.FDPGConfig(lam_warmup=0, hidden=32)
fp.install(HP, policy_cls=FakePolicy, config_cls=FakeConfig)


def make_batch(b=16, done_frac=0.25, seed=0):
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(b, 2 * NOBS, dtype=torch.bool)
    pad[: int(b * done_frac), -1] = True
    state = torch.randn(b, 2 * NOBS, SD, generator=g)
    state[:, :, 0] = pad[:, -1:].float() * 2 - 1     # near the end of an episode is visible in the state
    return {
        fp.OBS_STATE: state,
        fp.OBS_IMAGES: torch.randn(b, 2 * NOBS, 2, 3, 4, 4, generator=g),
        fp.ACTION: torch.rand(b, H, A, generator=g) * 2 - 1,
        f"{fp.OBS_STATE}_is_pad": pad,
    }


def test_delta_indices_extended():
    assert FakeConfig().observation_delta_indices == [-1, 0, 3, 4]


def test_init_adds_critic_and_target_frozen():
    pol = FakePolicy(FakeConfig())
    assert hasattr(pol, "fdpg_q") and hasattr(pol, "fdpg_v")
    assert all(not p.requires_grad for p in pol.fdpg_q_target.parameters())
    assert int(pol.fdpg_step) == 0
    assert "fdpg_step" in pol.state_dict()


def test_optim_groups_separate_critic_lr():
    pol = FakePolicy(FakeConfig())
    groups = pol.get_optim_params()
    crit = groups[-1]
    assert crit["lr"] == HP.critic_lr
    crit_ids = {id(p) for p in crit["params"]}
    want = {id(p) for p in list(pol.fdpg_q.parameters()) + list(pol.fdpg_v.parameters())}
    assert crit_ids == want
    other = {id(p) for g in groups[:-1] for p in g["params"]}
    assert not other & crit_ids
    assert not {id(p) for p in pol.fdpg_q_target.parameters()} & (other | crit_ids)


def test_forward_finite_and_logs():
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    loss, info = pol(make_batch())
    assert torch.isfinite(loss)
    assert info["fdpg_lambda"] == HP.lam_max
    assert info["fdpg_done_frac"] == pytest.approx(0.25)
    assert info["fdpg_grad_norm"] > 0


def test_actor_terms_do_not_train_critic():
    """Critic params must get gradients only from the critic loss."""
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    pol._fdpg_hp = fp.FDPGConfig(lam_warmup=0, hidden=32, critic_weight=0.0)
    loss, _ = pol(make_batch())
    loss.backward()
    for p in list(pol.fdpg_q.parameters()) + list(pol.fdpg_v.parameters()):
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_critic_does_not_train_encoder():
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    pol._fdpg_hp = fp.FDPGConfig(lam_warmup=0, hidden=32, mode="bc_critic", consistency_weight=0.0)
    # zero the actor loss path: only critic loss remains if we detach the actor
    batch = make_batch()
    loss, info = pol(batch)
    critic_only = torch.tensor(info["fdpg_v_loss"] + info["fdpg_q_loss"])
    assert critic_only > 0
    pol.zero_grad()
    # recompute critic loss alone through the same modules
    cur, nxt = fp.split_time(batch, NOBS)
    c = pol.observation_encoder.encode(cur).detach()
    v = pol.fdpg_v(c)
    v.sum().backward()
    assert all(p.grad is None for p in pol.observation_encoder.parameters())


def test_push_increases_q():
    """First-order check: moving a_hat along delta raises min(Q1, Q2)."""
    torch.manual_seed(1)
    q = fp.TwinQ(cond_dim=6, act_dim=K * A, hidden=32)
    cond = torch.randn(8, 6)
    a_hat = torch.randn(8, H, A)
    u = torch.randn(8, H, A)
    delta, _ = fp.critic_push(q, cond, a_hat, u, s=NOBS - 1, k=K, alpha=1e-3, eps=1e-6)
    s = NOBS - 1
    with torch.no_grad():
        before = torch.minimum(*q(cond, a_hat[:, s:s + K].reshape(8, -1)))
        after = torch.minimum(*q(cond, (a_hat + delta)[:, s:s + K].reshape(8, -1)))
    assert (after > before).all()
    # outside the executed slice nothing moves
    assert torch.count_nonzero(delta[:, :s]) == 0 and torch.count_nonzero(delta[:, s + K:]) == 0
    # step size: ||delta|| = alpha * ||u||
    assert torch.allclose(delta.flatten(1).norm(dim=1), 1e-3 * u.flatten(1).norm(dim=1), rtol=1e-3)


def test_lambda_schedule():
    hp = fp.FDPGConfig(lam_max=0.5, lam_warmup=100, lam_start=50)
    assert fp.lambda_at(0, hp) == 0.0
    assert fp.lambda_at(50, hp) == 0.0
    assert fp.lambda_at(100, hp) == pytest.approx(0.25)
    assert fp.lambda_at(1000, hp) == 0.5
    assert fp.lambda_at(1000, fp.FDPGConfig(mode="bc_critic")) == 0.0


def test_lambda_zero_is_bc_plus_consistency():
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    pol._fdpg_hp = fp.FDPGConfig(mode="bc_critic", hidden=32, critic_weight=0.0)
    loss, info = pol(make_batch())
    assert loss.item() == pytest.approx(info["fdpg_bc"] + info["fdpg_cons"], rel=1e-5)


def test_update_polyak_and_step():
    pol = FakePolicy(FakeConfig())
    with torch.no_grad():
        for p in pol.fdpg_q.parameters():
            p.add_(1.0)
    before = [tp.clone() for tp in pol.fdpg_q_target.parameters()]
    pol.update()
    assert int(pol.fdpg_step) == 1
    for b, tp, p in zip(before, pol.fdpg_q_target.parameters(), pol.fdpg_q.parameters()):
        assert torch.allclose(tp, 0.995 * b + 0.005 * p)


def test_split_time_rejects_unpatched_batch():
    b = make_batch()
    b[fp.OBS_STATE] = b[fp.OBS_STATE][:, :NOBS]
    with pytest.raises(RuntimeError):
        fp.split_time(b, NOBS)


def test_training_reduces_critic_error_and_matches_terminal_value():
    """Short loop: critic learns that terminal transitions are worth ~1, others ~gamma^K * V."""
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    pol._fdpg_hp = fp.FDPGConfig(hidden=32, lam_warmup=200, lam_start=200, critic_lr=1e-3)
    opt = torch.optim.Adam([{"params": g["params"], "lr": g.get("lr", 1e-3)} for g in pol.get_optim_params()])
    first = last = None
    for i in range(400):
        loss, info = pol(make_batch(b=32, seed=i))
        opt.zero_grad()
        loss.backward()
        opt.step()
        pol.update()
        if i == 0:
            first = info["fdpg_q_loss"]
        last = info["fdpg_q_loss"]
    assert last < first
    assert int(pol.fdpg_step) == 400
    assert info["fdpg_lambda"] == pytest.approx(fp.lambda_at(399, pol._fdpg_hp))
    # terminal transitions should be valued near the terminal reward (1.0), non-terminal lower
    b = make_batch(b=64, done_frac=0.5, seed=9999)
    cur, _ = fp.split_time(b, NOBS)
    with torch.no_grad():
        c = pol.observation_encoder.encode(cur)
        q = torch.minimum(*pol.fdpg_q(c, b[fp.ACTION][:, NOBS - 1:NOBS - 1 + K].reshape(64, -1)))
    done = b[f"{fp.OBS_STATE}_is_pad"][:, -1]
    assert q[done].mean() == pytest.approx(1.0, abs=0.15)
    assert q[~done].mean() < q[done].mean()


# ---------------------------------------------------------------- reward from data (online rollouts)

def test_reward_window_success_is_discounted_and_terminal():
    hp = fp.FDPGConfig(gamma=0.9)
    rew = torch.zeros(2, K, 1)
    rew[0, 2] = 1.0                                   # success at the 3rd executed step
    batch = {fp.REWARD: rew, f"{fp.REWARD}_is_pad": torch.zeros(2, K, dtype=torch.bool)}
    r, d, src = fp.chunk_reward_and_done(batch, 2, K, hp)
    assert src == "data"
    assert r.tolist() == pytest.approx([0.9 ** 2, 0.0])
    assert d.tolist() == [1.0, 0.0]


def test_reward_window_padding_is_ignored():
    hp = fp.FDPGConfig()
    rew = torch.ones(1, K, 1)                          # padded copies of a success frame
    pad = torch.ones(1, K, dtype=torch.bool)
    pad[0, 0] = False
    r, d, _ = fp.chunk_reward_and_done({fp.REWARD: rew, f"{fp.REWARD}_is_pad": pad}, 1, K, hp)
    assert r.item() == pytest.approx(1.0) and d.item() == 1.0


def test_reward_inferred_without_column():
    b = make_batch(b=4, done_frac=0.5)
    r, d, src = fp.chunk_reward_and_done(b, 4, K, fp.FDPGConfig())
    assert src == "inferred" and d.tolist() == [1.0, 1.0, 0.0, 0.0] and r.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_reward_delta_indices_patched():
    assert FakeConfig().reward_delta_indices == list(range(K))


def test_forward_uses_data_reward_when_present():
    torch.manual_seed(0)
    pol = FakePolicy(FakeConfig())
    b = make_batch(b=8, done_frac=0.0)
    rew = torch.zeros(8, K, 1)
    rew[:3, -1] = 1.0
    b[fp.REWARD] = rew
    _, info = pol(b)
    assert info["fdpg_reward_from_data"] == 1.0
    assert info["fdpg_done_frac"] == pytest.approx(3 / 8)


# ---------------------------------------------------------------- critic architecture (split vs concat)

def test_split_critic_has_separate_projections():
    q = fp.TwinQ(cond_dim=4112, act_dim=168, hidden=512, arch="split", proj_dim=256)
    w_obs = q.q1.obs[1].weight
    w_act = q.q1.act[0].weight
    assert w_obs.shape == (256, 4112) and w_act.shape == (256, 168)
    assert q.q1.mlp[0].weight.shape == (512, 512)       # [obs_proj, act_proj] -> hidden
    c, a = torch.randn(3, 4112), torch.randn(3, 168)
    q1, q2 = q(c, a)
    assert q1.shape == (3,) and q2.shape == (3,)


def test_concat_critic_keeps_old_parameter_names():
    q = fp.TwinQ(cond_dim=6, act_dim=4, hidden=8, arch="concat")
    keys = set(q.state_dict())
    assert "q1.1.weight" in keys and "q2.1.weight" in keys      # names of pre-split checkpoints
    assert q.q1[1].weight.shape == (8, 10)


@pytest.mark.parametrize("arch", ["split", "concat"])
def test_critic_settings_detected_from_checkpoint(tmp_path, arch):
    safetensors = pytest.importorskip("safetensors.torch")
    hp = fp.FDPGConfig(hidden=32, critic_arch=arch, proj_dim=16)
    pol = FakePolicy(FakeConfig())                       # installed with defaults; rebuild critic
    cond_dim = pol.observation_encoder.conditioning_dim
    pol.fdpg_q = fp.TwinQ(cond_dim, A * K, hp.hidden, arch=arch, proj_dim=hp.proj_dim)
    safetensors.save_file({k: v.contiguous() for k, v in pol.state_dict().items()},
                          str(tmp_path / "model.safetensors"))
    found = fp.critic_settings_from_checkpoint(tmp_path)
    assert found["critic_arch"] == arch and found["hidden"] == 32
    if arch == "split":
        assert found["proj_dim"] == 16


def test_no_critic_in_plain_checkpoint(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({"noise_predictor.w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    assert fp.critic_settings_from_checkpoint(tmp_path) == {}


def test_bad_arch_rejected():
    with pytest.raises(SystemExit):
        fp.FDPGConfig.from_args({"critic_arch": "huge"})
