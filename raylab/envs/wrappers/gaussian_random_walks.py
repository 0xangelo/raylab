"""Wrapper for introducing irrelevant state variables."""
import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin
from .mixins import RNGMixin
from .utils import assert_flat_box_space


class GaussianRandomWalks(IrrelevantRedundantMixin, RNGMixin, gym.ObservationWrapper):
    """Add gaussian random walk variables to the observations.

    Arguments:
        env: a gym environment instance
        size: the number of random walks to append to the observation.
        loc: mean of the Gaussian distribution
        scale: stddev of the Gaussian distribution
    """

    def __init__(self, env: gym.Env, size: int, loc: float = 0.0, scale: float = 1.0):
        assert_flat_box_space(env.observation_space, self)
        super().__init__(env)
        self._size = size
        self._loc = loc
        self._scale = scale
        self._random_walk = None

        original = self.env.observation_space
        low = np.concatenate([original.low, [-np.inf] * size])
        high = np.concatenate([original.high, [np.inf] * size])
        self.observation_space = Box(low=low, high=high, dtype=original.dtype)

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return self._size

    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        self._random_walk = self._random_walk + self.np_random.normal(
            loc=self._loc, scale=self._scale, size=self._size
        )
        return self._random_walk

    def reset(self, **kwargs):
        self._random_walk = self.np_random.normal(
            loc=self._loc, scale=self._scale, size=self._size
        )
        return super().reset(**kwargs)
