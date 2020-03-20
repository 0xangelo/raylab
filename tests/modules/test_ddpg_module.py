# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.ddpg_module import DDPGModule


@pytest.fixture(params=(True, False), ids=("Double Q", "Single Q"))
def double_q(request):
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
def full_config(
    double_q, exploration, exploration_gaussian_sigma, beta, smooth_target_policy
):
    return {
        "double_q": double_q,
        "exploration": exploration,
        "exploration_gaussian_sigma": exploration_gaussian_sigma,
        "smooth_target_policy": smooth_target_policy,
        "actor": {"beta": beta},
    }


@pytest.fixture
def module_batch_fn(module_and_batch_fn):
    return partial(module_and_batch_fn, DDPGModule)


def test_module_creation(module_batch_fn, full_config):
    module, _ = module_batch_fn(full_config)

    assert "actor" in module
    assert "critics" in module
    assert "target_critics" in module
    expected_n_critics = 2 if full_config["double_q"] else 1
    assert len(module.critics) == expected_n_critics


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


def test_target_critics(module_batch_fn, double_q):
    module, batch = module_batch_fn({"double_q": double_q})
    for mod in module.target_critics:
        val = mod(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32
