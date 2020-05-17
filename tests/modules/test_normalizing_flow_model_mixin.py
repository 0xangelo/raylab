# pylint: disable=missing-docstring,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-locals
from gym.spaces import Box
import numpy as np
import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.modules.mixins import NormalizingFlowModelMixin

from .utils import make_batch, make_module


class DummyModule(NormalizingFlowModelMixin, nn.ModuleDict):
    # pylint:disable=abstract-method
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_model(obs_space, action_space, config))


@pytest.fixture
def agent():
    return DummyModule


OBS_SPACES = tuple(Box(-np.inf, np.inf, shape=(s,)) for s in (2, 8))
RESIDUAL = (True, False)


@pytest.fixture(params=OBS_SPACES, ids=(repr(o) for o in OBS_SPACES))
def obs_space(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("StatefulPrior", "StatelessPrior"))
def conditional_prior(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("StatefulFlow", "StatelessFlow"))
def conditional_flow(request):
    return request.param


@pytest.fixture(params=RESIDUAL, ids=(f"Residual(f{r})" for r in RESIDUAL))
def residual(request):
    return request.param


@pytest.fixture
def config(conditional_prior, conditional_flow, residual):
    return {
        "model": {
            "residual": residual,
            "conditional_prior": conditional_prior,
            "conditional_flow": conditional_flow,
        }
    }


def test_sampler(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
    batch = make_batch(obs_space, action_space)

    obs = batch[SampleBatch.CUR_OBS]
    act = batch[SampleBatch.ACTIONS]
    sampler = module.model.rsample

    samples, logp = sampler(obs, act)
    samples_, _ = sampler(obs, act)
    assert samples.shape == obs.shape
    assert samples.dtype == obs.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_params(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
    batch = make_batch(obs_space, action_space)

    obs = batch[SampleBatch.CUR_OBS]
    act = batch[SampleBatch.ACTIONS]
    params = module.model(obs, act)
    assert "loc" in params
    assert "scale" in params
    loc, scale = params["loc"], params["scale"]

    assert loc.shape == obs.shape
    assert scale.shape == obs.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    if config["model"]["conditional_prior"]:
        prior_params = set(module.model.params.parameters())
        for par in prior_params:
            par.grad = None
        loc.mean().backward()
        assert any(p.grad is not None for p in prior_params)
        assert all(p.grad is None for p in set(module.parameters()) - prior_params)

        for par in prior_params:
            par.grad = None
        scale = module.model(obs, act)["scale"]
        scale.mean().backward()
        assert any(p.grad is not None for p in prior_params)
        assert all(p.grad is None for p in set(module.parameters()) - prior_params)


def test_reproduce(agent, obs_space, action_space, config, torch_script):
    module = make_module(agent, obs_space, action_space, config, torch_script)
    batch = make_batch(obs_space, action_space)

    obs = batch[SampleBatch.CUR_OBS]
    act = batch[SampleBatch.ACTIONS]
    new_obs = batch[SampleBatch.NEXT_OBS]
    obs_, logp_ = module.model.reproduce(obs, act, new_obs)
    assert obs_.shape == new_obs.shape
    assert obs_.dtype == new_obs.dtype
    assert torch.allclose(obs_, new_obs, atol=1e-5)
    assert logp_.shape == batch[SampleBatch.REWARDS].shape

    obs_.mean().backward()
    params = set(module.model.parameters())
    assert all(p.grad is not None for p in params)
    assert all(p.grad is None for p in set(module.parameters()) - params)
