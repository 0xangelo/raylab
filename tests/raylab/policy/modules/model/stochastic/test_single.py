import pytest
import torch
from torch.autograd import grad

from raylab.policy.modules.model.stochastic.single import MLPModel
from raylab.policy.modules.model.stochastic.single import ResidualStochasticModel
from raylab.torch.nn import NormalParams


@pytest.fixture(scope="module", params=(True, False), ids=lambda x: f"Residual({x})")
def residual(request):
    return request.param


@pytest.fixture
def module_cls():
    return MLPModel


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def spec(module_cls, input_dependent_scale):
    return module_cls.spec_cls(input_dependent_scale=input_dependent_scale)


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, residual):
    mod = module_cls(obs_space, action_space, spec)
    return ResidualStochasticModel(mod) if residual else mod


def test_init(module):
    assert hasattr(module, "params")
    assert hasattr(module, "dist")


def test_forward(mocker, module, obs, act, next_obs):
    # pylint:disable=too-many-arguments
    params_spy = mocker.spy(NormalParams, "forward")

    params = module(obs, act)
    assert "loc" in params
    assert "scale" in params
    assert params_spy.called

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

    module(obs, act)["scale"].mean().backward()
    assert any(p.grad is not None for p in params)


def test_sample(module, obs, act, next_obs, rew):
    sampler = module.sample
    inputs = (module(obs, act),)

    samples, logp = sampler(*inputs)
    samples_, _ = sampler(*inputs)
    assert samples.shape == next_obs.shape
    assert samples.dtype == next_obs.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)


def test_rsample_gradient_propagation(module, obs, act):
    sampler = module.rsample
    obs.requires_grad_(True)
    act.requires_grad_(True)
    params = module(obs, act)

    sample, logp = sampler(params)
    assert obs.grad_fn is None
    assert act.grad_fn is None
    sample.sum().backward(retain_graph=True)
    assert obs.grad is not None
    assert act.grad is not None

    obs.grad, act.grad = None, None
    logp.sum().backward()
    assert obs.grad is not None
    assert act.grad is not None


def test_log_prob(module, obs, act, next_obs, rew):
    logp = module.log_prob(next_obs, module(obs, act))

    assert torch.is_tensor(logp)
    assert logp.shape == rew.shape

    logp.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


def test_reproduce(module, obs, act, next_obs, rew):
    next_obs_, logp_ = module.reproduce(next_obs, module(obs, act))
    assert next_obs_.shape == next_obs.shape
    assert next_obs_.dtype == next_obs.dtype
    assert torch.allclose(next_obs_, next_obs, atol=1e-5)
    assert logp_.shape == rew.shape

    next_obs_.mean().backward()
    params = set(module.parameters())
    assert all(p.grad is not None for p in params)


def test_deterministic(module, obs, act, rew):
    params = module(obs, act)

    obs1, logp1 = module.deterministic(params)
    assert torch.is_tensor(obs1)
    assert torch.is_tensor(logp1)
    assert obs1.shape == obs.shape
    assert logp1.shape == rew.shape
    assert obs1.dtype == obs.dtype
    assert logp1.dtype == rew.dtype

    obs2, logp2 = module.deterministic(params)
    assert torch.allclose(obs1, obs2)
    assert torch.allclose(logp1, logp2)

    assert obs1.grad_fn is not None
    obs1.sum().backward()
    assert any([p.grad is not None for p in module.parameters()])


def test_script(module):
    torch.jit.script(module)


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_model_ograd(module, obs, act):
    model = torch.jit.script(module)
    obs = obs.clone().requires_grad_()

    rsample, _ = model.rsample(model(obs, act))
    (ograd,) = grad(rsample.mean(), [obs], create_graph=True)
    print(ograd)
    ograd.mean().backward()
    assert obs.grad is not None


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_model_agrad(module, obs, act):
    model = torch.jit.script(module)
    act = act.clone().requires_grad_()

    rsample, _ = model.rsample(model(obs, act))
    (agrad,) = grad(rsample.mean(), [act], create_graph=True)
    print(agrad)
    agrad.mean().backward()
    assert act.grad is not None
