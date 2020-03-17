# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import ActionOutput


@pytest.fixture
def args_kwargs():
    return (10, torch.ones(4).neg(), torch.ones(4)), dict(beta=1.2)


def test_module_creation(torch_script, args_kwargs):
    args, kwargs = args_kwargs
    module = ActionOutput(*args, **kwargs)
    if torch_script:
        module = torch.jit.script(module)

    inputs = torch.randn(2, args[0])
    module(inputs)
