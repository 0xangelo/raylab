import pytest
import torch
from torch.autograd import grad

from raylab.pytorch.nn import FullyConnected
from raylab.pytorch.nn import StateActionEncoder


@pytest.fixture(params=(1, 2, 4), ids=lambda x: f"InFeatures:{x}")
def in_features(request):
    return request.param


@pytest.fixture
def inputs(in_features):
    return torch.ones(1, in_features)


@pytest.fixture(params=((10,), (4, 4)), ids=lambda x: f"Units:{x}")
def units(request):
    return request.param


@pytest.fixture(params=(None, "Tanh", "ReLU", "ELU"))
def activation(request):
    return request.param


@pytest.fixture(params=(True, False), ids=lambda x: f"LayerNorm:{x}")
def layer_norm(request):
    return request.param


@pytest.fixture
def kwargs(units, activation, layer_norm):
    return dict(units=units, activation=activation, layer_norm=layer_norm)


@pytest.fixture
def fully_connected(in_features, kwargs, torch_script):
    module = FullyConnected(in_features=in_features, **kwargs)
    if torch_script:
        module = torch.jit.script(module)
    return module


def test_fully_connected(fully_connected, inputs):
    out = fully_connected(inputs)
    assert torch.is_tensor(out)
    assert out.grad_fn is not None
    out.mean().backward()
    assert all([p.grad is not None for p in fully_connected.parameters()])


@pytest.fixture
def fc_no_units(in_features, activation, layer_norm):
    return FullyConnected(
        in_features=in_features, units=(), activation=activation, layer_norm=layer_norm
    )


def test_fc_no_units(fc_no_units, inputs):
    assert not list(fc_no_units.parameters())

    out = fc_no_units(inputs)
    assert out is inputs


@pytest.fixture
def script_fc(in_features, kwargs):
    return torch.jit.script(FullyConnected(in_features=in_features, **kwargs))


def test_script_fc_grad(script_fc, inputs, activation, units):
    inputs = inputs.clone().requires_grad_(True)
    out = script_fc(inputs)
    (igrad,) = grad(out.sum(), [inputs], create_graph=True)
    assert torch.is_tensor(igrad)
    assert igrad.shape == inputs.shape

    if len(units) > 1 or activation != "ReLU":
        igrad.mean().backward()
        assert inputs.grad is not None


# ======================================================================================
# StateActionEncoder
# ======================================================================================


@pytest.fixture(params=(1, 4), ids=lambda x: f"ObsDim:{x}")
def obs_dim(request):
    return request.param


@pytest.fixture(params=(1, 3), ids=lambda x: f"ActDim:{x}")
def act_dim(request):
    return request.param


@pytest.fixture
def obs(obs_dim):
    return torch.randn(1, obs_dim)


@pytest.fixture
def act(act_dim):
    return torch.randn(1, act_dim)


@pytest.fixture(params=(True, False), ids=lambda x: f"DelayAction:{x}")
def delay_action(request):
    return request.param


@pytest.fixture
def sae_kwargs(obs_dim, act_dim, delay_action, kwargs):
    return dict(
        obs_dim=obs_dim, action_dim=act_dim, delay_action=delay_action, **kwargs
    )


@pytest.fixture
def sae(sae_kwargs, torch_script):
    module = StateActionEncoder(**sae_kwargs)
    if torch_script:
        module = torch.jit.script(module)
    return module


def test_state_action_encoder(sae, obs, act):
    out = sae(obs, act)
    assert torch.is_tensor(out)

    out.mean().backward()
    assert all([p.grad is not None for p in sae.parameters()])


@pytest.fixture
def script_sae(obs_dim, act_dim, delay_action, layer_norm):
    return torch.jit.script(
        StateActionEncoder(
            obs_dim=obs_dim,
            action_dim=act_dim,
            delay_action=delay_action,
            units=(32, 32),
            activation="Swish",
            layer_norm=layer_norm,
        )
    )


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_sae_ograd(script_sae, obs, act):
    print(script_sae.code)
    print(script_sae.obs_module)
    print(script_sae.sequential_module.code)
    # print(script_sae.sequential_module.sequential.code)
    obs = obs.clone().requires_grad_(True)

    out = script_sae(obs, act)
    (ograd,) = grad(out.mean(), [obs], create_graph=True)
    assert torch.is_tensor(ograd)

    ograd.mean().backward()
    assert obs.grad is not None


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_sae_agrad(script_sae, obs, act):
    act = act.clone().requires_grad_(True)

    out = script_sae(obs, act)
    (agrad,) = grad(out.mean(), [act], create_graph=True)
    assert torch.is_tensor(agrad)

    agrad.mean().backward()
    assert act.grad is not None
