# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture(scope="module", params=(True, False), ids=lambda x: f"Residual({x})")
def module_cls(request):
    from raylab.policy.modules.model.stochastic.single import MLPModel
    from raylab.policy.modules.model.stochastic.single import ResidualMLPModel

    return ResidualMLPModel if request.param else MLPModel


@pytest.fixture
def spec(module_cls):
    return module_cls.spec_cls()


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, input_dependent_scale):
    return module_cls(obs_space, action_space, spec, input_dependent_scale)


def test_sample(module, batch):
    new_obs = batch[SampleBatch.NEXT_OBS]
    sampler = module.rsample
    inputs = (batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])

    samples, logp = sampler(*inputs)
    samples_, _ = sampler(*inputs)
    assert samples.shape == new_obs.shape
    assert samples.dtype == new_obs.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_params(module, batch):
    inputs = (batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
    new_obs = batch[SampleBatch.NEXT_OBS]

    params = module(*inputs)
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    assert loc.shape == new_obs.shape
    assert scale.shape == new_obs.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    params = set(module.parameters())
    for par in params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in params)

    for par in params:
        par.grad = None
    module(*inputs)["scale"].mean().backward()
    assert any(p.grad is not None for p in params)


def test_log_prob(module, batch):
    logp = module.log_prob(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )

    assert torch.is_tensor(logp)
    assert logp.shape == batch[SampleBatch.REWARDS].shape

    logp.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


def test_reproduce(module, batch):
    obs, act, new_obs = [
        batch[k]
        for k in (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)
    ]

    new_obs_, logp_ = module.reproduce(obs, act, new_obs)
    assert new_obs_.shape == new_obs.shape
    assert new_obs_.dtype == new_obs.dtype
    assert torch.allclose(new_obs_, new_obs, atol=1e-5)
    assert logp_.shape == batch[SampleBatch.REWARDS].shape

    new_obs_.mean().backward()
    params = set(module.parameters())
    assert all(p.grad is not None for p in params)


def test_script(module):
    torch.jit.script(module)
