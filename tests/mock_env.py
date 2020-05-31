"""Dummy gym.Env subclasses."""
import gym
import numpy as np
import torch
from gym.spaces import Box
from ray.rllib.utils import override

from raylab.envs.rewards import register as register_reward
from raylab.envs.rewards import RewardFn
from raylab.envs.termination import register as register_termination
from raylab.envs.termination import TerminationFn


class MockEnv(gym.Env):  # pylint: disable=abstract-method
    """Dummy environment with continuous action space."""

    def __init__(self, _):
        self.horizon = 200
        self.time = 0

        low = np.array([-1] * 3 + [0], dtype=np.float32)
        high = np.array([1] * 4, dtype=np.float32)
        self.observation_space = Box(low=low, high=high)

        action_dim = 3
        self.action_space = Box(high=1, low=-1, shape=(action_dim,), dtype=np.float32)

        self.goal = torch.zeros(self.observation_space.shape)
        self.state = None

    @override(gym.Env)
    def reset(self):
        self.time = 0
        self.state = self.observation_space.sample()
        self.state[-1] = 0
        return self.state

    @override(gym.Env)
    def step(self, action):
        self.time += 1
        self.state[:3] = np.clip(
            self.state[:3] + action,
            self.observation_space.low[:3],
            self.observation_space.high[:3],
        )
        self.state[-1] = self.time / self.horizon
        reward = np.linalg.norm((self.state - self.goal.numpy()), axis=-1)
        return self.state, reward, self.time >= self.horizon, {}

    def reward_fn(self, state, action, next_state):
        # pylint: disable=missing-docstring,unused-argument
        return torch.norm(next_state - self.goal, dim=-1)


@register_reward("MockEnv")
class MockReward(RewardFn):  # pylint:disable=missing-class-docstring
    @override(RewardFn)
    def forward(self, state, action, next_state):
        return torch.norm(next_state, p=2, dim=-1)


@register_termination("MockEnv")
class MockTermination(TerminationFn):  # pylint:disable=missing-class-docstring
    def __init__(self, _):
        super().__init__()

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return next_state[..., -1] >= 1.0
