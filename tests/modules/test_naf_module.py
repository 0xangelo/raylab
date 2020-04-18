# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch

from raylab.modules.naf_module import NAFModule


@pytest.fixture(params=(True, False), ids=("Double Q", "Single Q"))
def double_q(request):
    return request.param


@pytest.fixture
def config(double_q):
    return {"double_q": double_q}


@pytest.fixture
def module_batch_fn(module_and_batch_fn):
    return partial(module_and_batch_fn, NAFModule)


def test_module_creation(module_batch_fn, config):
    module, _ = module_batch_fn(config)
    assert isinstance(module, (torch.nn.ModuleDict, torch.jit.ScriptModule))
