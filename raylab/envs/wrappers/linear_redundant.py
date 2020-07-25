# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import IrrelevantRedundantMixin
from .mixins import RNGMixin
from .utils import assert_flat_box_space


class LinearRedundant(IrrelevantRedundantMixin, RNGMixin, gym.ObservationWrapper):
    """Adds linear redundant variables to the environment's observations.

    Computes redundant variables as a linear function, with weight matrix
    :math:`W`, of the observation features. Samples :math:`W` upon reset from a
    uniform distribution :math:`Unif_{d\times d}(0, 1)`, where :math:`d` is the
    observation size.

    Args:
        env: Gym environment instance
    """

    def __init__(self, env: gym.Env):
        assert_flat_box_space(env.observation_space, self)
        super().__init__(env)
        self._wmat: np.ndarray = None

        original = self.env.observation_space
        size = original.shape[0]
        low = np.concatenate([original.low, [-np.inf] * size]).astype(original.dtype)
        high = np.concatenate([original.high, [np.inf] * size]).astype(original.dtype)
        self.observation_space = Box(low=low, high=high)

        self._set_reward_if_possible()
        self._set_termination_if_possible()

    @property
    def added_size(self):
        return self.env.observation_space.shape[0]

    def reset(self, **kwargs) -> np.ndarray:
        size = self.env.observation_space.shape[0]
        self._wmat = self.np_random.uniform(low=0.0, high=1.0, size=(size, size))
        return super().reset(**kwargs)

    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        return self._wmat @ observation

    @staticmethod
    def wrap_env_function(func: callable,) -> callable:
        """Wrap base env reward/termination function to ignore added variables.

        Args:
            func: Callable for reward/termination function
        """
        # pylint:disable=arguments-differ
        def env_fn(state, action, next_state):
            size = state.size(-1) // 2
            return func(state[..., :-size], action, next_state[..., :-size])

        return env_fn
