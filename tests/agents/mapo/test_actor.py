# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


GRAD_ESTIMATOR = "SF PD".split()


@pytest.fixture(params=GRAD_ESTIMATOR)
def grad_estimator(request):
    return request.param


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, grad_estimator):
    return policy_and_batch_fn({"grad_estimator": grad_estimator})


def test_next_action_grads_propagation(policy_and_batch):
    policy, batch = policy_and_batch
    obs = batch[SampleBatch.CUR_OBS]

    acts = policy.module.actor(obs)
    torch.manual_seed(42)
    surrogate = policy.one_step_action_value_surrogate(obs, acts)
    surrogate.mean().backward()
    grads = [p.grad.clone() for p in policy.module.actor.parameters()]

    policy.module.actor.zero_grad()

    acts = policy.module.actor(obs)
    fix_acts = acts.detach().requires_grad_()
    torch.manual_seed(42)
    surrogate = policy.one_step_action_value_surrogate(obs, fix_acts)
    surrogate.mean().backward()
    acts.backward(gradient=fix_acts.grad)
    grads_ = [p.grad.clone() for p in policy.module.actor.parameters()]

    assert all(torch.allclose(g, g_) for g, g_ in zip(grads, grads_))


def test_actor_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.madpg_loss(batch)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    params = list(policy.module.actor.parameters())
    assert all(p.grad is not None for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert all(not torch.isnan(p.grad).any() for p in params)
    assert all(not torch.allclose(p.grad, torch.zeros_like(p)) for p in params)
