import pytest
from torch import nn


@pytest.fixture(scope="module")
def optim_cls():
    from raylab.torch.optim.kfac import KFAC

    return KFAC


@pytest.fixture
def module():
    return nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 4))


@pytest.fixture
def optim(optim_cls, module):
    return optim_cls(module, eps=1e-3)


def test_all_linear_registered(optim, module):
    linear_params = set(
        p for m in module.modules() if isinstance(m, nn.Linear) for p in m.parameters()
    )
    optim_params = set(p for group in optim.param_groups for p in group["params"])

    assert not linear_params.symmetric_difference(optim_params)
