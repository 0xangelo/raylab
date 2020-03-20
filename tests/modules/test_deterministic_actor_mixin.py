# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import DDPGModule, MAPOModule


@pytest.fixture(params=(DDPGModule, MAPOModule))
def module_cls(request):
    return request.param


@pytest.fixture(params=(None, "gaussian", "parameter_noise"))
def exploration(request):
    return request.param


@pytest.fixture(params=(0.3, 0.0))
def exploration_gaussian_sigma(request):
    return request.param


@pytest.fixture(params=(0.8, 1.2))
def beta(request):
    return request.param


@pytest.fixture(
    params=(True, False), ids=("Smooth Target Policy", "Hard Target Policy")
)
def smooth_target_policy(request):
    return request.param


@pytest.fixture
def full_config(exploration, exploration_gaussian_sigma, beta, smooth_target_policy):
    return {
        "exploration": exploration,
        "exploration_gaussian_sigma": exploration_gaussian_sigma,
        "smooth_target_policy": smooth_target_policy,
        "actor": {"beta": beta},
    }


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_module_creation(module_batch_fn, full_config):
    module, _ = module_batch_fn(full_config)

    assert "actor" in module
    actor = module.actor
    assert "policy" in module.actor
    assert "behavior" in module.actor
    assert "target_policy" in module.actor
    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(actor.policy.parameters(), actor.target_policy.parameters())
    )


def test_policy(module_batch_fn, beta):
    module, batch = module_batch_fn({"actor": {"beta": beta}})
    action_dim = batch[SampleBatch.ACTIONS][0].numel()

    policy_out = module.actor.policy(batch[SampleBatch.CUR_OBS])
    norms = policy_out.norm(p=1, dim=-1, keepdim=True) / action_dim
    assert policy_out.shape[-1] == action_dim
    assert policy_out.dtype == torch.float32
    assert (norms <= (beta + torch.finfo(torch.float32).eps)).all()


def test_behavior(module_batch_fn, exploration, exploration_gaussian_sigma):
    module, batch = module_batch_fn(
        {
            "exploration": exploration,
            "exploration_gaussian_sigma": exploration_gaussian_sigma,
        }
    )
    action_dim = batch[SampleBatch.ACTIONS][0].numel()

    samples = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    samples_ = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    assert samples.shape[-1] == action_dim
    assert samples.dtype == torch.float32
    assert not (
        (exploration == "gaussian" and exploration_gaussian_sigma != 0)
        and torch.allclose(samples, samples_)
    )
