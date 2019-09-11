# pylint: disable=missing-docstring,redefined-outer-name
import pytest
import torch.nn as nn
from gym import spaces

from raylab.algorithms.naf.naf_policy import NAFTorchPolicy


@pytest.fixture(params=[None, "trace", "script"])
def torch_script(request):
    return request.param


@pytest.fixture(params=[None, "diag_gaussian", "full_gaussian"])
def exploration(request):
    return request.param


@pytest.fixture
def config(torch_script, exploration):
    return {
        "module": {"layers": [32, 32], "activation": "elu", "ortho_init_gain": 1.0},
        "torch_script": torch_script,
        "exploration": exploration,
    }


@pytest.fixture
def obs_space():
    return spaces.Box(low=-1, high=1, shape=(4,))


@pytest.fixture
def action_space():
    return spaces.Box(low=-1, high=1, shape=(2,))


def test_make_module(obs_space, action_space, config):
    # pylint: disable=protected-access
    module = NAFTorchPolicy._make_module(obs_space, action_space, config)
    assert isinstance(module, nn.ModuleDict)
