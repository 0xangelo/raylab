import math

import pytest
import torch

from raylab.policy.losses import DAPO
from raylab.policy.losses import MAPO
from raylab.policy.losses.abstract import Loss
from raylab.policy.losses.mixins import EnvFunctionsMixin


@pytest.fixture
def mapo(models, stochastic_policy, action_critics, reward_fn, termination_fn):
    critics, _ = action_critics
    loss_fn = MAPO(models, stochastic_policy, critics)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


def test_mapo_init(mapo):
    assert isinstance(mapo, Loss)
    assert isinstance(mapo, EnvFunctionsMixin)

    assert hasattr(mapo, "gamma")
    assert hasattr(mapo, "alpha")
    assert hasattr(mapo, "model_samples")
    assert hasattr(mapo, "grad_estimator")
    assert hasattr(mapo, "_modules")
    assert "models" in mapo._modules
    assert "policy" in mapo._modules
    assert "critics" in mapo._modules
    assert hasattr(mapo, "_rng")

    mapo.seed(42)
    assert hasattr(mapo, "_rng")

    assert mapo.initialized


def test_mapo_call(mapo, batch, models, obs, act):
    tensor, info = mapo(batch)
    assert torch.is_tensor(tensor)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, (float, int)) for v in info.values())
    assert "loss(actor)" in info
    assert "entropy" in info

    dist_params = models(obs, act)
    assert all(["model_mean_" + k in info for k in dist_params])


def test_mapo_compile(mapo, batch):
    mapo.compile()
    mapo(batch)


@pytest.fixture
def dynamics_fn():
    def func(obs, act):
        loc = obs + act.mean(dim=-1, keepdim=True)
        sample = torch.randn_like(obs) + loc
        log_prob = torch.sum(
            -((sample - loc) ** 2) / 2
            - torch.ones_like(obs).log()
            - math.log(math.sqrt(2 * math.pi)),
            dim=-1,
        )
        return sample, log_prob

    return func


@pytest.fixture
def dapo(dynamics_fn, stochastic_policy, action_critics, reward_fn, termination_fn):
    critics, _ = action_critics
    loss_fn = DAPO(dynamics_fn, stochastic_policy, critics)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


def test_dapo_init(dapo):
    assert isinstance(dapo, Loss)
    assert isinstance(dapo, EnvFunctionsMixin)

    assert hasattr(dapo, "gamma")
    assert hasattr(dapo, "alpha")
    assert hasattr(dapo, "grad_estimator")
    assert hasattr(dapo, "dynamics_fn")
    assert hasattr(dapo, "_modules")
    assert "policy" in dapo._modules
    assert "critics" in dapo._modules

    assert dapo.initialized


def test_dapo_call(dapo, batch):
    tensor, info = dapo(batch)
    assert torch.is_tensor(tensor)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, (float, int)) for v in info.values())
    assert "loss(actor)" in info
    assert "entropy" in info


def test_dapo_compile(dapo, batch):
    dapo.compile()
    dapo(batch)
