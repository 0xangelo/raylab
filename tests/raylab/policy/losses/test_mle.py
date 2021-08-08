import pytest
import torch
from nnrl.nn.model import SME

from raylab.policy.losses import MaximumLikelihood


@pytest.fixture
def loss_fn(models):
    return MaximumLikelihood(models)


def test_init(loss_fn, models):
    assert hasattr(loss_fn, "batch_keys")
    assert hasattr(loss_fn, "models")
    assert isinstance(loss_fn.models, SME)
    assert all([m is n for m, n in zip(models, loss_fn.models)])


def test_call(loss_fn, batch):
    loss, info = loss_fn(batch)

    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert isinstance(info, dict)
    assert all([isinstance(k, str) for k in info.keys()])
    assert all([isinstance(v, float) for v in info.values()])

    loss.sum().backward()
    assert all(
        [any([p.grad is not None for p in m.parameters()]) for m in loss_fn.models]
    )


def test_compile(loss_fn, batch):
    loss_fn.compile()
    loss, _ = loss_fn(batch)

    assert torch.is_tensor(loss)
    loss.sum().backward()
