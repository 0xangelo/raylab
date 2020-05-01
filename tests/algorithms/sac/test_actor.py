# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture(params=(True, False))
def input_dependent_scale(request):
    return {"module": {"actor": {"input_dependent_scale": request.param}}}


@pytest.fixture(params=(True, False))
def clipped_double_q(request):
    return {"clipped_double_q": request.param}


def test_actor_loss(policy_and_batch_fn, clipped_double_q, input_dependent_scale):
    policy, batch = policy_and_batch_fn({**clipped_double_q, **input_dependent_scale})
    loss, _ = policy.compute_actor_loss(batch, policy.module, policy.config)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert all(p.grad is not None for p in policy.module.actor.parameters())
    assert all(p.grad is not None for p in policy.module.alpha.parameters())
    assert all(p.grad is not None for p in policy.module.critics.parameters())
