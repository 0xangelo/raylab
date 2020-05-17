# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.modules.mixins import StateValueMixin


class DummyModule(StateValueMixin, nn.ModuleDict):
    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_critic(obs_space, action_space, config))


@pytest.fixture(params=(DummyModule,))
def module_cls(request):
    return request.param


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


TARGET_VF = (True, False)


@pytest.fixture(params=TARGET_VF, ids=(f"TargetVF({v})" for v in TARGET_VF))
def target_vf(request):
    return request.param


@pytest.fixture
def config(target_vf):
    return {"critic": {"target_vf": target_vf}}


def test_module_creation(module_batch_fn, config):
    module, batch = module_batch_fn(config)
    config = config["critic"]

    assert "critic" in module
    assert ("target_critic" in module) == config["target_vf"]

    critics = (module.critic,)
    if config["target_vf"]:
        critics = critics + (module.target_critic,)
    vals = [m(batch[SampleBatch.CUR_OBS]) for m in critics]
    for val in vals:
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32

    if config["target_vf"]:
        assert all(
            torch.allclose(p, p_)
            for p, p_ in zip(critics[0].parameters(), critics[1].parameters())
        )
