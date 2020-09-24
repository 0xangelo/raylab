import pytest
import torch

from raylab.policy.losses import ActionDPG
from raylab.policy.losses import DeterministicPolicyGradient
from raylab.policy.losses import ReparameterizedSoftPG


@pytest.fixture
def stochastic_actor(stochastic_policy):
    return stochastic_policy


@pytest.fixture
def critics(action_critics):
    return action_critics[0]


@pytest.fixture
def soft_pg_loss(stochastic_actor, critics, alpha_module):
    return ReparameterizedSoftPG(
        actor=stochastic_actor, critic=critics, alpha=alpha_module
    )


def test_soft_pg_loss(soft_pg_loss, stochastic_actor, critics, batch, obs):
    loss, info = soft_pg_loss(batch)
    actor = stochastic_actor

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())
    assert all(p.grad is not None for p in critics.parameters())

    assert "loss(actor)" in info
    assert "entropy" in info
    dist_params = actor(obs)
    keys = tuple(k for k, v in dist_params.items() if v.requires_grad)
    prefix = "policy/"
    assert all([prefix + "mean_" + k in info for k in keys])
    assert all([prefix + "max_" + k in info for k in keys])
    assert all([prefix + "min_" + k in info for k in keys])


@pytest.fixture
def deterministic_actor(deterministic_policies):
    policy, _ = deterministic_policies
    return policy


@pytest.fixture(params=(None, 40), ids=lambda x: f"dQdaClip({x})")
def dqda_clipping(request):
    return request.param


@pytest.fixture(params=(True, False), ids=lambda x: f"ClipNorm({x})")
def clip_norm(request):
    return request.param


@pytest.fixture
def action_dpg_loss(deterministic_actor, critics, dqda_clipping, clip_norm):
    loss_fn = ActionDPG(deterministic_actor, critics)
    loss_fn.dqda_clipping = dqda_clipping
    loss_fn.clip_norm = clip_norm
    return loss_fn


def test_acme_dpg(action_dpg_loss, deterministic_actor, critics, batch):
    loss, info = action_dpg_loss(batch)
    actor = deterministic_actor

    assert torch.is_tensor(loss)
    assert loss.shape == ()

    loss.backward()
    assert all([p.grad is not None for p in actor.parameters()])
    assert all([p.grad is None for p in critics.parameters()])

    assert isinstance(info, dict)
    assert "loss(actor)" in info
    assert "dqda_norm" in info


def test_dpg_grad_equivalence(deterministic_actor, critics, batch):
    actor = deterministic_actor
    default_dpg = DeterministicPolicyGradient(actor, critics)
    acme_dpg = ActionDPG(actor, critics)

    loss_default, _ = default_dpg(batch)
    loss_acme, _ = acme_dpg(batch)

    default_grad = torch.autograd.grad(loss_default, actor.parameters())
    acme_grad = torch.autograd.grad(loss_acme, actor.parameters())

    zip_grads = list(zip(default_grad, acme_grad))
    assert all([torch.allclose(d, a) for d, a in zip_grads])
