# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import SVGModule


@pytest.fixture(params=(SVGModule,))
def module_cls(request):
    return request.param


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_module_creation(module_batch_fn):
    module, batch = module_batch_fn({})

    assert "critic" in module
    assert "target_critic" in module

    critic, target_critic = module.critic, module.target_critic
    vals = [m(batch[SampleBatch.CUR_OBS]) for m in (critic, target_critic)]
    for val in vals:
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32

    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(critic.parameters(), target_critic.parameters())
    )
