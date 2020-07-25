# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin
from .mixins import RNGMixin
from .utils import assert_flat_box_space


class CorrelatedIrrelevant(IrrelevantRedundantMixin, RNGMixin, gym.ObservationWrapper):
    """Add correlated random variables to the environment's observations.

    Samples :math:`Uniform(0, 1)` random variables on reset and appends them to
    the observation. For every subsequent timestep t, appends the initial
    variables, exponentiated to the power of t, to the observation.

    Args:
        env: Gym environment
        size: Number of random reward-irrelevant variables
    """

    def __init__(self, env: gym.Env, size: int):
        assert_flat_box_space(env.observation_space, self)
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

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return self._size

    def reset(self, **kwargs):
        self._uvars = self.np_random.uniform(size=self._size)
        self._timestep = 0
        return super().reset(**kwargs)

    def step(self, action):
        self._timestep += 1
        return super().step(action)

    def _added_vars(self, _) -> np.ndarray:
        return self._uvars ** self._timestep
