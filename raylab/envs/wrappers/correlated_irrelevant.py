# pylint:disable=missing-module-docstring
from typing import List
from typing import Optional

import gym
import gym.utils.seeding as seeding
import numpy as np
from gym.spaces import Box

from .utils import assert_box_observation_space


class CorrelatedIrrelevant(gym.ObservationWrapper):
    """Add correlated random variables to the environment's observations.

    Samples :math:`Uniform(0, 1)` random variables on reset and appends them to
    the observation. For every subsequent timestep t, appends the initial
    variables, exponentiated to the power of t, to the observation.

    Args:
        env: Gym environment
        size: Number of random reward-irrelevant variables
    """

    def __init__(self, env: gym.Env, size: int):
        assert_box_observation_space(env, self)
        super().__init__(env)
        self._size = size
        self._timestep: int = 0
        self._uvars: np.ndarray = None

        original = self.observation_space
        low = np.concatenate([original.low, [0] * self._size])
        high = np.concatenate([original.high, [1] * self._size])
        self.observation_space = Box(
            low=low.astype(original.dtype), high=high.astype(original.dtype)
        )

        self.np_random, _ = seeding.np_random()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seeds = super().seed(seed) or []
        self.np_random, seed_ = seeding.np_random(seed)
        return seeds + [seed_]

    def reset(self, **kwargs):
        self._uvars = self.np_random.uniform(size=self._size)
        self._timestep = 0
        return super().reset(**kwargs)

    def step(self, action):
        self._timestep += 1
        return super().step(action)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        irrelevant = self._uvars ** self._timestep
        observation = np.concatenate([observation, irrelevant])
        return observation.astype(self.observation_space.dtype)
