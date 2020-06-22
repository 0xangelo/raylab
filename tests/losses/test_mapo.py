# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.losses import MAPO
from raylab.losses.abstract import Loss


@pytest.fixture
def loss_fn(models, stochastic_policy, action_critics, reward_fn, termination_fn):
    critics, _ = action_critics
    loss_fn = MAPO(models, stochastic_policy, critics)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


def test_init(loss_fn):
    assert isinstance(loss_fn, Loss)

    assert hasattr(loss_fn, "gamma")
    assert hasattr(loss_fn, "model_samples")
    assert hasattr(loss_fn, "grad_estimator")
    assert hasattr(loss_fn, "_modules")
    assert "models" in loss_fn._modules
    assert "policy" in loss_fn._modules
    assert "critics" in loss_fn._modules
    assert hasattr(loss_fn, "_rng")

    loss_fn.seed(42)
    assert hasattr(loss_fn, "_rng")

    assert hasattr(loss_fn, "_env")
    assert loss_fn._env.initialized
    assert loss_fn.initialized


def test_call(loss_fn, batch):
    tensor, info = loss_fn(batch)
    assert torch.is_tensor(tensor)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, (float, int)) for v in info.values())
    assert "loss(actor)" in info
    assert "entropy" in info


def test_compile(loss_fn, batch):
    loss_fn.compile()
    assert isinstance(loss_fn._modules, torch.jit.ScriptModule)
    loss_fn(batch)
