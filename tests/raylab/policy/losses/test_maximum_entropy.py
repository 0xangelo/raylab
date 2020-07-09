import pytest
import torch

from raylab.policy.losses import MaximumEntropyDual


@pytest.fixture
def target_entropy():
    return -4


@pytest.fixture
def alpha():
    from raylab.policy.modules.actor.policy.stochastic import Alpha

    return Alpha(0.05)


@pytest.fixture
def loss_fn(alpha, stochastic_policy, target_entropy):
    return MaximumEntropyDual(alpha, stochastic_policy.sample, target_entropy)


def test_loss(loss_fn, batch, alpha):
    loss, info = loss_fn(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert "loss(alpha)" in info
    assert "curr_alpha" in info

    loss.backward()
    assert all(p.grad is not None for p in alpha.parameters())
