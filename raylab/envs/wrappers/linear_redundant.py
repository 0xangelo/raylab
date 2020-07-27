# pylint:disable=missing-module-docstring
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin
from .mixins import RNGMixin
from .utils import assert_flat_box_space
from .utils import check_redundant_size_compat


class LinearRedundant(IrrelevantRedundantMixin, RNGMixin, gym.ObservationWrapper):
    """Adds linear redundant variables to the environment's observations.

    Computes redundant variables as a linear function, with weight matrix
    :math:`W`, of the observation features. Samples :math:`W` upon reset from a
    uniform distribution :math:`Unif_{d\times d}(0, 1)`, where :math:`d` is the
    observation size.

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

        self._wmat: np.ndarray = None

        low = np.concatenate([original.low, [-np.inf] * size]).astype(original.dtype)
        high = np.concatenate([original.high, [np.inf] * size]).astype(original.dtype)
        self.observation_space = Box(low=low, high=high)

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return self.env.observation_space.shape[0]

    def reset(self, **kwargs) -> np.ndarray:
        size = self._size
        self._wmat = self.np_random.uniform(low=0.0, high=1.0, size=(size, size))
        return super().reset(**kwargs)

    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        return self._wmat @ observation[: self._size]
