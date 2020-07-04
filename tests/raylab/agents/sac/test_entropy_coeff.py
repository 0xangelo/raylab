# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture(params=("auto", -4))
def config(request):
    return {"target_entropy": request.param}


def test_alpha_loss(policy_and_batch_fn, config):
    policy, batch = policy_and_batch_fn(config)
    loss, info = policy.loss_alpha(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert "loss(alpha)" in info
    assert "curr_alpha" in info

    loss.backward()
    assert all(p.grad is not None for p in policy.module.alpha.parameters())
