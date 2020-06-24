# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import copy
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from raylab.agents.mapo import MAPOTorchPolicy
from raylab.agents.sac import SACTorchPolicy
from raylab.losses import MAPO
from raylab.losses import SoftCDQLearning
from raylab.losses import SPAML
from raylab.policy import EnvFnMixin
from raylab.policy import ModelTrainingMixin
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


@pytest.fixture(scope="module")
def sample_batch(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=32)


@pytest.fixture(scope="module")
def policy(obs_space, action_space, reward_fn, termination_fn):
    policy = MAPOTorchPolicy(obs_space, action_space, {})
    policy.set_reward_from_callable(reward_fn)
    policy.set_termination_from_callable(termination_fn)
    return policy


def test_init(policy):
    assert isinstance(policy, SACTorchPolicy)
    assert isinstance(policy, EnvFnMixin)
    assert isinstance(policy, ModelTrainingMixin)

    assert hasattr(policy.module, "models")
    assert hasattr(policy.module, "actor")
    assert hasattr(policy.module, "critics")
    assert hasattr(policy.module, "alpha")

    assert isinstance(policy.loss_model, SPAML)
    assert isinstance(policy.loss_actor, MAPO)
    assert isinstance(policy.loss_critic, SoftCDQLearning)


def test_learn_on_batch(policy, sample_batch):
    modules = nn.ModuleList([policy.module.actor, policy.module.critics])
    old_params = copy.deepcopy(list(modules.parameters()))

    info = policy.learn_on_batch(sample_batch)
    assert "loss(actor)" in info
    assert "loss(critics)" in info
    assert "entropy" in info
    assert "grad_norm(actor)" in info
    assert "grad_norm(critics)" in info
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, (int, float)) for v in info.values())
    assert all(np.isfinite(v) for v in info.values())

    new_params = list(modules.parameters())
    assert all(not torch.allclose(o, n) for o, n in zip(old_params, new_params))


def test_compile(policy):
    with mock.patch("raylab.losses.MAPO.compile") as mapo, mock.patch(
        "raylab.losses.SPAML.compile"
    ) as spaml:
        policy.compile()
        assert isinstance(policy.module, torch.jit.ScriptModule)
        assert mapo.called
        assert spaml.called
