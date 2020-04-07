# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import DDPGModule, MAPOModule


@pytest.fixture(params=(DDPGModule, MAPOModule))
def module_cls(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("DoubleQ", "SingleQ"))
def double_q(request):
    return request.param


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_module_creation(module_batch_fn, double_q):
    module, batch = module_batch_fn({"double_q": double_q})

    assert "critics" in module
    assert "target_critics" in module
    expected_n_critics = 2 if double_q else 1
    assert len(module.critics) == expected_n_critics

    critics, target_critics = module.critics, module.target_critics
    vals = [
        m(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        for critics in (critics, target_critics)
        for m in critics
    ]
    for val in vals:
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32

    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(critics.parameters(), target_critics.parameters())
    )
