# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import numpy as np
import torch.nn as nn

from raylab.algorithms.naf.naf_policy import NAFTorchPolicy


@pytest.fixture(params=[None, "diag_gaussian", "full_gaussian", "parameter_noise"])
def exploration(request):
    return request.param


@pytest.fixture(params=[True, False])
def clipped_double_q(request):
    return request.param


@pytest.fixture
def config(exploration, clipped_double_q):
    return {
        "module": {
            "units": (32, 32),
            "activation": "ELU",
            "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
        },
        "exploration": exploration,
        "clipped_double_q": clipped_double_q,
    }


def test_make_module(obs_space, action_space, config):
    policy = NAFTorchPolicy(obs_space, action_space, config)
    assert isinstance(policy.module, nn.ModuleDict)
