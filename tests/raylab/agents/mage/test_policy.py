import numpy as np
import pytest
import torch

from raylab.agents.mage import MAGETorchPolicy
from raylab.policy.losses import DeterministicPolicyGradient
from raylab.policy.losses import MAGE
from raylab.policy.losses import MaximumLikelihood
from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def reward_fn():
    def func(obs, act, new_obs):
        return new_obs[..., 0] - obs[..., 0] - act.norm(dim=-1)

    return func


@pytest.fixture(scope="module")
def termination_fn():
    def func(obs, *_):
        return torch.randn_like(obs[..., 0]) > 0

    return func


@pytest.fixture
def policy_cls():
    return MAGETorchPolicy


def test_default_config(policy_cls):
    defaults = policy_cls.options.defaults
    assert "polyak" in defaults
    assert "model_training" in defaults
    assert "model_sampling" not in defaults


@pytest.fixture
def policy(policy_cls, obs_space, action_space, reward_fn, termination_fn):
    policy = policy_cls(obs_space, action_space, {})
    policy.set_reward_from_callable(reward_fn)
    policy.set_termination_from_callable(termination_fn)
    return policy


def test_init(policy):
    assert hasattr(policy, "module")

    module = policy.module
    assert hasattr(module, "models")
    assert hasattr(module, "actor")
    assert hasattr(module, "target_actor")
    assert hasattr(module, "critics")
    assert hasattr(module, "target_critics")

    assert hasattr(policy, "optimizers")

    optimizers = policy.optimizers
    assert "models" in optimizers
    assert "actor" in optimizers
    assert "critics" in optimizers

    assert hasattr(policy, "loss_model")
    assert hasattr(policy, "loss_actor")
    assert hasattr(policy, "loss_critic")
    assert isinstance(policy.loss_model, MaximumLikelihood)
    assert isinstance(policy.loss_actor, DeterministicPolicyGradient)
    assert isinstance(policy.loss_critic, MAGE)


@pytest.fixture
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


def test_learn_on_batch(policy, samples):
    info = policy.learn_on_batch(samples)
    assert all(
        f"train_loss(models[{i}])" not in info for i in range(len(policy.module.models))
    )
    assert "learner" not in info
    assert "learner_stats" not in info
    assert "loss(actor)" in info
    assert "loss(critics)" in info
    assert "grad_norm(actor)" in info
    assert "grad_norm(critics)" in info
    assert "grad_norm(models)" not in info

    assert np.isfinite(info["loss(actor)"])
    assert np.isfinite(info["loss(critics)"])
    assert np.isfinite(info["grad_norm(actor)"])
    assert np.isfinite(info["grad_norm(critics)"])


def test_compile(policy, mocker):
    method = mocker.spy(MAGE, "compile")
    policy.compile()
    assert isinstance(policy.module, torch.jit.ScriptModule)
    assert method.called
