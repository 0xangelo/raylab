# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.modules.mixins import ActionValueMixin


class DummyModule(ActionValueMixin, nn.ModuleDict):
    # pylint:disable=abstract-method
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_critic(obs_space, action_space, config))


@pytest.fixture(params=(DummyModule,))
def module_cls(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("DoubleQ", "SingleQ"))
def double_q(request):
    return request.param


@pytest.fixture
def config(double_q):
    return {"critic": {"double_q": double_q}}


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_module_creation(module_batch_fn, config):
    module, batch = module_batch_fn(config)
    double_q = config["critic"]["double_q"]

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
