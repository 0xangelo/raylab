# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_policy):
    return policy_and_batch_fn(svg_policy, {})


def test_policy_reproduce(policy_and_batch):
    policy, batch = policy_and_batch

    acts = batch[SampleBatch.ACTIONS]
    _acts = policy.module.policy_reproduce(batch[SampleBatch.CUR_OBS], acts)
    assert _acts.shape == acts.shape
    assert _acts.dtype == acts.dtype
    assert torch.allclose(_acts, acts, atol=1e-6)

    _acts.mean().backward()
    pi_params = set(policy.module.policy.parameters())
    assert all(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - pi_params)


def test_model_reproduce(policy_and_batch):
    policy, batch = policy_and_batch

    next_obs = batch[SampleBatch.NEXT_OBS]
    _next_obs = policy.module.model_reproduce(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], next_obs
    )
    assert _next_obs.shape == next_obs.shape
    assert _next_obs.dtype == next_obs.dtype
    assert torch.allclose(_next_obs, next_obs, atol=1e-6)

    _next_obs.mean().backward()
    model_params = set(policy.module.model.parameters())
    assert all(p.grad is not None for p in model_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - model_params)
