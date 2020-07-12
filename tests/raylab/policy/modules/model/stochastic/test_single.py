import pytest
import torch


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


def test_sample(module, obs, act, next_obs, rew):
    sampler = module.sample
    inputs = (obs, act)

    samples, logp = sampler(*inputs)
    samples_, _ = sampler(*inputs)
    assert samples.shape == next_obs.shape
    assert samples.dtype == next_obs.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)


def test_rsample_gradient_propagation(module, obs, act):
    sampler = module.rsample
    act.requires_grad_(True)

    sample, logp = sampler(obs, act)
    assert obs.grad_fn is None
    assert act.grad_fn is None
    sample.sum().backward(retain_graph=True)
    assert obs.grad is not None
    assert act.grad is not None

    obs.grad, act.grad = None, None
    logp.sum().backward()
    assert obs.grad is not None
    assert act.grad is not None


def test_params(module, obs, act, next_obs):
    inputs = (obs, act)

    params = module(*inputs)
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    assert loc.shape == next_obs.shape
    assert scale.shape == next_obs.shape
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


def test_log_prob(module, obs, act, next_obs, rew):
    logp = module.log_prob(obs, act, next_obs)

    assert torch.is_tensor(logp)
    assert logp.shape == rew.shape

    logp.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


def test_reproduce(module, obs, act, next_obs, rew):
    next_obs_, logp_ = module.reproduce(obs, act, next_obs)
    assert next_obs_.shape == next_obs.shape
    assert next_obs_.dtype == next_obs.dtype
    assert torch.allclose(next_obs_, next_obs, atol=1e-5)
    assert logp_.shape == rew.shape

    next_obs_.mean().backward()
    params = set(module.parameters())
    assert all(p.grad is not None for p in params)


def test_script(module):
    torch.jit.script(module)
