# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.utils.tracking_dict import UsageTrackingDict

from raylab.utils.debug import fake_batch
from raylab.utils.pytorch import convert_to_tensor


@pytest.fixture(
    params=(pytest.param(True, marks=pytest.mark.slow), False),
    ids=("TorchScript", "Eager"),
    scope="module",
)
def torch_script(request):
    return request.param


@pytest.fixture(scope="module")
def module_and_batch_fn(obs_space, action_space, torch_script):
    def make_module_and_batch(module_cls, config):
        config["torch_script"] = torch_script
        module = module_cls(obs_space, action_space, config)

        batch = UsageTrackingDict(fake_batch(obs_space, action_space, batch_size=10))
        batch.set_get_interceptor(partial(convert_to_tensor, device="cpu"))

        return torch.jit.script(module) if torch_script else module, batch

    return make_module_and_batch
