# pylint: disable=missing-docstring,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-locals
from gym.spaces import Box
import pytest
from ray.rllib import SampleBatch
import torch

from raylab.modules.catalog import TRPOTang2018

from .utils import make_batch, make_module


ACTION_SPACES = (Box(-1, 1, shape=(2,)), Box(-1, 1, shape=(3,)))


@pytest.fixture(params=ACTION_SPACES, ids=(repr(a) for a in ACTION_SPACES))
def action_space(request):
    return request.param


@pytest.fixture(params=(TRPOTang2018,))
def nf_agent(request):
    return request.param


def test_sampler(nf_agent, obs_space, action_space, torch_script):
    module = make_module(nf_agent, obs_space, action_space, {}, torch_script)
    batch = make_batch(obs_space, action_space)
    action = batch[SampleBatch.ACTIONS]

    sampler = module.actor.rsample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)

    logp_ = module.actor.log_prob(batch[SampleBatch.CUR_OBS], samples)
    assert torch.allclose(logp, logp_, atol=1e-5)


def test_flow_params(nf_agent, obs_space, action_space, torch_script):
    module = make_module(nf_agent, obs_space, action_space, {}, torch_script,)
    batch = make_batch(obs_space, action_space)
    params = module.actor(batch[SampleBatch.CUR_OBS])
    assert "state" in params


def test_reproduce(nf_agent, obs_space, action_space, torch_script):
    module = make_module(nf_agent, obs_space, action_space, {}, torch_script)
    batch = make_batch(obs_space, action_space)

    acts = batch[SampleBatch.ACTIONS]
    acts_, logp_ = module.actor.reproduce(batch[SampleBatch.CUR_OBS], acts)
    assert acts_.shape == acts.shape
    assert acts_.dtype == acts.dtype
    assert torch.allclose(acts_, acts, atol=1e-5)
    assert logp_.shape == batch[SampleBatch.REWARDS].shape

    acts_.mean().backward()
    pi_params = set(module.actor.parameters())
    assert all(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)
