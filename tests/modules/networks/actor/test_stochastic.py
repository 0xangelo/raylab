# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from gym.spaces import Box
from gym.spaces import Discrete
from ray.rllib import SampleBatch

from raylab.utils.debug import fake_batch

pytest.skip(reason="Not implemented")


@pytest.fixture(scope="module")
def module_cls():
    from raylab.modules.networks.actor.stochastic import StochasticActor

    return StochasticActor


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def cont_spec(module_cls, input_dependent_scale):
    return module_cls.spec_cls(input_dependent_scale=input_dependent_scale)


@pytest.fixture
def spec(module_cls):
    return module_cls.spec_cls()


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


@pytest.fixture
def disc_module(module_cls, obs_space, disc_space, spec, torch_script):
    mod = module_cls(obs_space, disc_space, spec)
    return torch.jit.script(mod) if torch_script else mod


@pytest.fixture
def cont_module(module_cls, obs_space, cont_space, spec, torch_script):
    mod = module_cls(obs_space, cont_space, spec)
    return torch.jit.script(mod) if torch_script else mod


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, torch_script):
    mod = module_cls(obs_space, action_space, spec)
    return torch.jit.script(mod) if torch_script else mod


@pytest.fixture
def disc_batch(obs_space, disc_space):
    samples = fake_batch(obs_space, disc_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def cont_batch(obs_space, cont_space):
    samples = fake_batch(obs_space, cont_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


def test_discrete_sampler(disc_module, disc_batch):
    module, batch = disc_module, disc_batch
    action = batch[SampleBatch.ACTIONS]

    sampler = module.actor.sample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_continuous_sampler(cont_module, cont_batch):
    module = cont_module
    batch = cont_batch
    action = batch[SampleBatch.ACTIONS]

    sampler = module.actor.rsample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_discrete_params(disc_module, disc_batch):
    module, batch = disc_module, disc_batch

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


def test_continuous_params(cont_module, cont_batch):
    module, batch = cont_module, cont_batch
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


def test_reproduce(cont_module, cont_batch):
    module, batch = cont_module, cont_batch

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
