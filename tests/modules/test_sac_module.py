# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.sac_module import SACModule


@pytest.fixture(params=(True, False))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("Double Q", "Single Q"))
def double_q(request):
    return request.param


@pytest.fixture
def module_batch_fn(module_and_batch_fn):
    return partial(module_and_batch_fn, SACModule)


def test_actor_params(module_batch_fn, input_dependent_scale):
    module, batch = module_batch_fn(
        {"actor": {"input_dependent_scale": input_dependent_scale}}
    )

    params = module.actor.params(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale_diag" in params

    loc, scale_diag = params.values()
    action_dim = batch[SampleBatch.ACTIONS][0].numel()
    assert loc.shape[-1] == action_dim
    assert scale_diag.shape[-1] == action_dim
    assert loc.dtype == torch.float32
    assert scale_diag.dtype == torch.float32

    pi_params = set(module.actor.policy.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)

    for par in pi_params:
        par.grad = None
    module.actor.params(batch[SampleBatch.CUR_OBS])["scale_diag"].mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)


def test_critic(module_batch_fn, double_q):
    module, batch = module_batch_fn({"double_q": double_q})
    vals = [
        m(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        for critics in (module.critics, module.target_critics)
        for m in critics
    ]
    for val in vals:
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32
