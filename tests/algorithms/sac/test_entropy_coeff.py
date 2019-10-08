import pytest
import torch


@pytest.fixture(params=(None, -4))
def target_entropy(request):
    return {"target_entropy": request.param}


def test_alpha_init(policy_and_batch_fn, target_entropy):
    policy, _ = policy_and_batch_fn(target_entropy)
    target = target_entropy["target_entropy"] or -len(policy.action_space.shape)

    assert policy.config["target_entropy"] is not None
    assert policy.config["target_entropy"] == target


def test_alpha_loss(policy_and_batch_fn, target_entropy):
    policy, batch = policy_and_batch_fn(target_entropy)
    loss, _ = policy.compute_alpha_loss(batch, policy.module, policy.config)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert policy.module.log_alpha.grad is not None
