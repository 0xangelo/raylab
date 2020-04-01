# pylint:disable=missing-docstring,redefined-outer-name,protected-access
from typing import Dict

import pytest
import torch
import torch.nn as nn

from raylab.modules import StateActionEncoder
from raylab.modules.flows import CondAffine1DHalfFlow


class MyModule(nn.Module):
    def __init__(self, in_size, out_size, encoder):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(in_size + encoder.out_features, out_size)

    def forward(self, inputs, cond: Dict[str, torch.Tensor]):
        # pylint:disable=arguments-differ
        encoded = self.encoder(cond["state"], cond["action"])
        return self.linear(torch.cat([inputs, encoded], dim=-1))


MLP_KWARGS = {
    "units": (24,) * 3,
    "activation": {"name": "LeakyReLU", "options": {"negative_slope": 0.2}},
}


def module_fn(kwargs):
    kwargs = kwargs.copy()

    def func(in_size, cond1_size, cond2_size, out_size):
        nonlocal kwargs
        encoder = StateActionEncoder(cond1_size, cond2_size, **kwargs)
        return MyModule(in_size, out_size, encoder)

    return func


@pytest.fixture(params=(MLP_KWARGS, None))
def scale_module(request):
    return module_fn(request.param) if request.param else lambda *x: None


@pytest.fixture(params=(MLP_KWARGS, None))
def shift_module(request):
    return module_fn(request.param) if request.param else lambda *x: None


@pytest.fixture(params=(True, False))
def parity(request):
    return request.param


@pytest.fixture
def module(parity, scale_module, shift_module, torch_script):
    def make_module(state_size, action_size):
        in_size = state_size - (state_size // 2)
        out_size = state_size // 2
        if parity:
            in_size, out_size = out_size, in_size
        scale = scale_module(in_size, state_size, action_size, out_size)
        shift = shift_module(in_size, state_size, action_size, out_size)
        mod = CondAffine1DHalfFlow(parity, scale, shift)
        return torch.jit.script(mod) if torch_script else mod

    return make_module


@pytest.fixture(params=(2, 4, 7))
def state_size(request):
    return request.param


@pytest.fixture(params=(2, 4, 7))
def action_size(request):
    return request.param


@pytest.fixture(params=((), (1,), (4,)))
def inputs(request, state_size, action_size):
    batch_shape = request.param
    input_ = torch.randn(batch_shape + (state_size,)).requires_grad_()
    cond = {
        "state": torch.randn_like(input_).requires_grad_(),
        "action": torch.randn(batch_shape + (action_size,)).requires_grad_(),
    }
    return input_, cond


def test_affine_half(module, inputs):
    inputs, cond = inputs
    module = module(cond["state"].size(-1), cond["action"].size(-1))

    scale = bool(list(module.scale.parameters()))
    shift = bool(list(module.shift.parameters()))

    latent, log_det = module(inputs, cond)
    if scale:
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    latent.sum().backward()
    assert inputs.grad is not None
    if scale or shift:
        assert all(cond[k].grad is not None for k in cond)

    latent = latent.detach().requires_grad_()

    input_, log_det = module(latent, cond, reverse=True)
    assert torch.allclose(input_, inputs, atol=1e-7)
    if scale:
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    input_.sum().backward()
    assert latent.grad is not None
