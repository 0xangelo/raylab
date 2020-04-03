# pylint: disable=missing-docstring,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-locals
import pytest
import torch
from gym.spaces import Box, Discrete
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import SACModule, SVGModule, TRPOModule

from .utils import make_batch, make_module

DISC_SPACES = (Discrete(2), Discrete(8))
CONT_SPACES = (Box(-1, 1, shape=(1,)), Box(-1, 1, shape=(3,)))
ACTION_SPACES = CONT_SPACES + DISC_SPACES


@pytest.fixture(params=DISC_SPACES, ids=(repr(a) for a in DISC_SPACES))
def disc_space(request):
    return request.param


@pytest.fixture(params=CONT_SPACES, ids=(repr(a) for a in CONT_SPACES))
def cont_space(request):
    return request.param


@pytest.fixture(params=ACTION_SPACES, ids=(repr(a) for a in ACTION_SPACES))
def action_space(request):
    return request.param


@pytest.fixture(params=(SACModule, SVGModule, TRPOModule))
def continuous_agent(request):
    return request.param


@pytest.fixture(params=(TRPOModule))
def discrete_agent(request):
    return request.param


@pytest.fixture(params=(SACModule, SVGModule, TRPOModule))
def agent(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("InputDepScale", "InputIndepScale"))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("MeanAction", "SampleAction"))
def mean_action_only(request):
    return request.param


def test_discrete_sampler(
    discrete_agent,
    obs_space,
    disc_space,
    input_dependent_scale,
    mean_action_only,
    torch_script,
):
    module = make_module(
        discrete_agent,
        obs_space,
        disc_space,
        {
            "mean_action_only": mean_action_only,
            "actor": {"input_dependent_scale": input_dependent_scale},
        },
        torch_script,
    )
    batch = make_batch(obs_space, disc_space)
    action = batch[SampleBatch.ACTIONS]

    sampler = module.actor.sample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert mean_action_only or not torch.allclose(samples, samples_)


def test_continuous_sampler(
    continuous_agent, obs_space, cont_space, mean_action_only, torch_script,
):
    module = make_module(
        continuous_agent,
        obs_space,
        cont_space,
        {"actor": {"input_dependent_scale": input_dependent_scale}},
        torch_script,
    )
    batch = make_batch(obs_space, cont_space)
    action = batch[SampleBatch.ACTIONS]

    sampler = module.actor.rsample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert mean_action_only or not torch.allclose(samples, samples_)


def test_discrete_params(discrete_agent, obs_space, disc_space, torch_script):
    module = make_module(discrete_agent, obs_space, disc_space, {}, torch_script)
    batch = make_batch(obs_space, disc_space)

    params = module.actor(batch[SampleBatch.CUR_OBS])
    assert "logits" in params
    logits = params["logits"]
    assert logits.shape[-1] == disc_space.n

    pi_params = set(module.actor.parameters())
    for par in pi_params:
        par.grad = None
    logits.mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)


def test_continuous_params(
    continuous_agent, obs_space, cont_space, input_dependent_scale, torch_script
):
    module = make_module(
        continuous_agent,
        obs_space,
        cont_space,
        {"actor": {"input_dependent_scale": input_dependent_scale}},
        torch_script,
    )
    batch = make_batch(obs_space, cont_space)
    params = module.actor(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    action = batch[SampleBatch.ACTIONS]
    assert loc.shape == action.shape
    assert scale.shape == action.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    pi_params = set(module.actor.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)

    for par in pi_params:
        par.grad = None
    module.actor(batch[SampleBatch.CUR_OBS])["scale"].mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)


def test_reproduce(continuous_agent, obs_space, cont_space, torch_script):
    module = make_module(continuous_agent, obs_space, cont_space, {}, torch_script)
    batch = make_batch(obs_space, cont_space)

    acts = batch[SampleBatch.ACTIONS]
    _acts = module.actor.reproduce(batch[SampleBatch.CUR_OBS], acts)
    assert _acts.shape == acts.shape
    assert _acts.dtype == acts.dtype
    assert torch.allclose(_acts, acts, atol=1e-6)

    _acts.mean().backward()
    pi_params = set(module.actor.parameters())
    assert all(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)
