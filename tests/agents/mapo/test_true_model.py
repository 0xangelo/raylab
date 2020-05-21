# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.utils.debug import fake_batch


@pytest.fixture
def config():
    return {"true_model": True, "env": "Navigation", "grad_estimator": "PD"}


@pytest.fixture
def navigation_env(envs):
    return envs["Navigation"]


@pytest.fixture
def policy_and_env(mapo_policy, navigation_env, config):
    env = navigation_env({})
    policy = mapo_policy(env.observation_space, env.action_space, config)
    policy.set_transition_kernel(env.transition_fn)
    return policy, env


def test_model_output(policy_and_env):
    policy, env = policy_and_env
    obs = policy.observation_space.sample()[None]
    act = policy.action_space.sample()[None]
    obs, act = map(policy.convert_to_tensor, (obs, act))
    obs, act = map(lambda x: x.requires_grad_(True), (obs, act))

    torch.manual_seed(42)
    sample, logp = policy.transition(obs, act)
    torch.manual_seed(42)
    next_obs, log_prob = env.transition_fn(obs, act)
    assert torch.allclose(sample, next_obs)
    assert torch.allclose(logp, log_prob)
    assert sample.grad_fn is not None
    assert logp.grad_fn is not None


def test_madpg_loss(policy_and_env):
    policy, _ = policy_and_env
    batch = policy._lazy_tensor_dict(
        fake_batch(policy.observation_space, policy.action_space, batch_size=10)
    )

    loss, info = policy.madpg_loss(batch)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.grad_fn is not None

    policy.module.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and not torch.isnan(p.grad).all()
        for p in policy.module.actor.parameters()
    )
