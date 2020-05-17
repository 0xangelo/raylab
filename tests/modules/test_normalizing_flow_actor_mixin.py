# pylint: disable=missing-docstring,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-locals
import pytest
import torch
import torch.nn as nn
from gym.spaces import Box
from ray.rllib import SampleBatch

from raylab.modules.mixins import NormalizingFlowActorMixin

from .utils import make_batch, make_module


class DummyModule(NormalizingFlowActorMixin, nn.ModuleDict):
    # pylint:disable=abstract-method
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_actor(obs_space, action_space, config))


@pytest.fixture
def agent():
    return DummyModule


ACTION_SPACES = (Box(-1, 1, shape=(2,)), Box(-1, 1, shape=(3,)))


@pytest.fixture(params=ACTION_SPACES, ids=(repr(a) for a in ACTION_SPACES))
def action_space(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("StateCondPrior", "StatelessPrior"))
def conditional_prior(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("StateCondFlow", "StatelessFlow"))
def conditional_flow(request):
    return request.param


@pytest.fixture
def config(conditional_prior, conditional_flow):
    return {
        "actor": {
            "conditional_prior": conditional_prior,
            "conditional_flow": conditional_flow,
        }
    }


def test_sampler(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
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


def test_params(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
    batch = make_batch(obs_space, action_space)

    params = module.actor(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale" in params
    loc, scale = params["loc"], params["scale"]

    action = batch[SampleBatch.ACTIONS]
    assert loc.shape == action.shape
    assert scale.shape == action.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    if config["actor"]["conditional_prior"]:
        prior_params = set(module.actor.params.parameters())
        for par in prior_params:
            par.grad = None
        loc.mean().backward()
        assert any(p.grad is not None for p in prior_params)
        assert all(p.grad is None for p in set(module.parameters()) - prior_params)

        for par in prior_params:
            par.grad = None
        scale = module.actor(batch[SampleBatch.CUR_OBS])["scale"]
        scale.mean().backward()
        assert any(p.grad is not None for p in prior_params)
        assert all(p.grad is None for p in set(module.parameters()) - prior_params)


def test_reproduce(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
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
