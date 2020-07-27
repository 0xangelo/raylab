# pylint:disable=missing-module-docstring
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin
from .utils import assert_flat_box_space
from .utils import check_redundant_size_compat


class NonlinearRedundant(IrrelevantRedundantMixin, gym.ObservationWrapper):
    r"""Adds nonlinear redundant variables to the environment's observations.

    Computes redundant variables as nonlinear functions.
    .. math::
        (x_t, \cos(x_t), \sin(x_t)) \in \mathbb{R}^{3d}

    Args:
        env: Gym environment instance
        size: Number of left-most features from the observation to use in
            computing redundant variables. Defaults to the observation size
    """

    def __init__(self, env: gym.Env, size: Optional[int] = None):
        assert_flat_box_space(env.observation_space, self)
        super().__init__(env)
        original = self.env.observation_space
        self._size = size = size or original.shape[0]
        check_redundant_size_compat(size, original)

        low = np.concatenate([original.low, [-1] * 2 * size]).astype(original.dtype)
        high = np.concatenate([original.high, [1] * 2 * size]).astype(original.dtype)
        self.observation_space = Box(low=low, high=high)

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return 2 * self.env.observation_space.shape[0]

    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        cos = np.cos(observation[: self._size])
        sin = np.sin(observation[: self._size])
        return np.concatenate([cos, sin])

    @staticmethod
    def wrap_env_function(func: callable, size: int) -> callable:
        return IrrelevantRedundantMixin.wrap_env_function(func, size * 2)
