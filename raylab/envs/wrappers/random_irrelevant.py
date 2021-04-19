# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin, RNGMixin
from .utils import assert_flat_box_space


class RandomIrrelevant(IrrelevantRedundantMixin, RNGMixin, gym.ObservationWrapper):
    """Add Normal random variables to the environment's observations.

    Args:
        env: Gym environment
        size: Number of random reward-irrelevant variables
        loc: Normal mean
        scale: Normal standard deviation
    """

    def __init__(self, env: gym.Env, size: int, loc: float = 0.0, scale: float = 1.0):
        assert_flat_box_space(env.observation_space, self)
        super().__init__(env)
        self._size = size
        self._loc = loc
        self._scale = scale

        original = self.observation_space
        low = np.concatenate([original.low, [-np.inf] * self._size])
        high = np.concatenate([original.high, [np.inf] * self._size])
        self.observation_space = Box(
            low=low.astype(original.dtype), high=high.astype(original.dtype)
        )

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return self._size

    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        return self.np_random.normal(loc=self._loc, scale=self._scale, size=self._size)
