# pylint:disable=missing-docstring,redefined-outer-name,protected-access,invalid-name
# pylint:disable=too-many-locals,too-many-arguments
from typing import Dict

import pytest
import torch
import torch.nn as nn

from raylab.modules import StateActionEncoder
from raylab.modules.flows import CondAffine1DHalfFlow


MLP_KWARGS = {
    "units": (24,) * 3,
    "activation": {"name": "LeakyReLU", "options": {"negative_slope": 0.2}},
}
PARITY = (False, True)
STATE_SHAPE = ((10, 2), (10, 4), (10, 7))
ACTION_SHAPE = ((10, 2), (10, 4), (10, 7))
SCALE = (True, False)
SHIFT = (True, False)


class MyMLP(nn.Module):
    def __init__(self, parity, state_size, action_size):
        super().__init__()
        out_size = state_size // 2
        in_size = state_size - out_size
        if parity:
            in_size, out_size = out_size, in_size
        self.encoder = StateActionEncoder(state_size, action_size, **MLP_KWARGS)
        self.linear = nn.Linear(in_size + self.encoder.out_features, out_size)

    def forward(self, inputs, cond: Dict[str, torch.Tensor]):
        # pylint:disable=arguments-differ
        state, action = cond["state"], cond["action"]
        encoded = self.encoder(state, action)
        return self.linear(torch.cat([inputs, encoded], dim=-1))


@pytest.fixture(params=PARITY, ids=(f"Parity({p})" for p in PARITY))
def parity(request):
    return request.param


@pytest.fixture(params=STATE_SHAPE, ids=(f"State({s})" for s in STATE_SHAPE))
def state(request):
    return torch.randn(request.param, requires_grad=True)


@pytest.fixture(params=ACTION_SHAPE, ids=(f"Action({s})" for s in ACTION_SHAPE))
def action(request):
    return torch.randn(request.param, requires_grad=True)


@pytest.fixture(params=SCALE, ids=(f"Scale({s})" for s in SCALE))
def scale(request):
    return request.param


@pytest.fixture(params=SHIFT, ids=(f"Shift({s})" for s in SHIFT))
def shift(request):
    return request.param


def test_cond_affine_half(parity, state, action, scale, shift, torch_script):
    z = torch.randn_like(state, requires_grad=True)
    state_size = state.size(-1)
    action_size = action.size(-1)

    scale_mod = MyMLP(parity, state_size, action_size) if scale else None
    shift_mod = MyMLP(parity, state_size, action_size) if shift else None
    mod = CondAffine1DHalfFlow(parity, scale_mod, shift_mod)
    module = torch.jit.script(mod) if torch_script else mod

    cond = dict(state=state, action=action)
    x, log_det = module(z, cond)
    if list(module.scale.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    x.sum().backward()
    assert z.grad is not None

    x = x.detach().requires_grad_()

    z_, log_det = module(x, cond, reverse=True)
    assert torch.allclose(z_, z, atol=1e-7)
    if list(module.scale.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    z_.sum().backward()
    assert x.grad is not None
