import pytest
import torch

from raylab.policy.losses import ReparameterizedSoftPG


@pytest.fixture
def actor(stochastic_policy):
    return stochastic_policy


@pytest.fixture
def critics(action_critics):
    return action_critics[0]


@pytest.fixture
def soft_pg_loss(actor, critics):
    return ReparameterizedSoftPG(actor, critics)


def test_soft_pg_loss(soft_pg_loss, actor, critics, batch):
    loss, info = soft_pg_loss(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())
    assert all(p.grad is not None for p in critics.parameters())

    assert "loss(actor)" in info
    assert "entropy" in info
