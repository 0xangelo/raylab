# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.losses import ModelEnsembleMLE


@pytest.fixture
def loss_fn(models):
    return ModelEnsembleMLE(models)


def test_init(loss_fn):
    assert hasattr(loss_fn, "batch_keys")


def test_call(loss_fn, batch):
    loss, info = loss_fn(batch)

    assert torch.is_tensor(loss)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, float) for v in info.values())


def test_compile(loss_fn, batch):
    loss_fn.compile()
    loss, _ = loss_fn(batch)

    assert torch.is_tensor(loss)
