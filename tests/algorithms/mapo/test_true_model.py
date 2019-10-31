# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture
def policy_and_env(mapo_policy, navigation_env):
    env = navigation_env({})
    policy = mapo_policy(env.observation_space, env.action_space, {"true_model": True})
    policy.set_reward_fn(env.reward_fn)
    policy.set_transition_fn(env.transition_fn)
    return policy, env


def test_model_output(policy_and_env):
    policy, env = policy_and_env
    obs = policy.observation_space.sample()[None]
    act = policy.action_space.sample()[None]
    obs, act = map(policy.convert_to_tensor, (obs, act))
    obs, act = map(lambda x: x.requires_grad_(True), (obs, act))

    torch.manual_seed(42)
    sample, logp = policy.module.model_sampler(obs, act)
    torch.manual_seed(42)
    next_obs, log_prob = env.transition_fn(obs, act)
    assert torch.allclose(sample, next_obs)
    assert torch.allclose(logp, log_prob)
    assert sample.grad_fn is None
    assert logp.grad_fn is not None
