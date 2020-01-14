# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture(params=(None, -4))
def config(request):
    return {"target_entropy": request.param}


def test_alpha_init(policy_and_batch_fn, config):
    policy, _ = policy_and_batch_fn(config)
    target = config["target_entropy"] or -policy.action_space.shape[0]

    assert policy.config["target_entropy"] is not None
    assert policy.config["target_entropy"] == target


def test_alpha_loss(policy_and_batch_fn, config):
    policy, batch = policy_and_batch_fn(config)
    loss, _ = policy.compute_alpha_loss(batch, policy.module, policy.config)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert policy.module.log_alpha.grad is not None
