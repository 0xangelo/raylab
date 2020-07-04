# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture(params=(True, False))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False))
def double_q(request):
    return request.param


def test_actor_loss(policy_and_batch_fn, double_q, input_dependent_scale):
    policy, batch = policy_and_batch_fn(
        {
            "module": {
                "actor": {"input_dependent_scale": input_dependent_scale},
                "critic": {"double_q": double_q},
            }
        }
    )
    loss, info = policy.loss_actor(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert all(p.grad is not None for p in policy.module.actor.parameters())
    assert all(p.grad is not None for p in policy.module.critics.parameters())

    assert "loss(actor)" in info
    assert "entropy" in info
