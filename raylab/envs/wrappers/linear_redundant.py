# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import RNGMixin
from .utils import assert_box_observation_space


class LinearRedundant(RNGMixin, gym.ObservationWrapper):
    """Adds linear redundant variables to the environment's observations.

    Computes redundant variables as a linear function, with weight matrix
    :math:`W`, of the observation features. Samples :math:`W` upon reset from a
    uniform distribution :math:`Unif_{d\times d}(0, 1)`, where :math:`d` is the
    observation size.

    Args:
        env: Gym environment instance
    """

    def __init__(self, env: gym.Env):
        assert_box_observation_space(env, self)
        super().__init__(env)
        self._wmat = None

        original = self.env.observation_space
        size = original.shape[0]
        low = np.concatenate([original.low, [-np.inf] * size]).astype(original.dtype)
        high = np.concatenate([original.high, [np.inf] * size]).astype(original.dtype)
        self.observation_space = Box(low=low, high=high)

    def reset(self, **kwargs) -> np.ndarray:
        size = self.env.observation_space.shape[0]
        self._wmat = self.np_random.uniform(low=0.0, high=1.0, size=(size, size))
        return super().reset(**kwargs)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        redundant = self._wmat @ observation
        observation = np.concatenate([observation, redundant])
        return observation.astype(self.observation_space.dtype)

    @staticmethod
    def wrap_env_function(func: callable) -> callable:
        """Wrap base env reward/termination function to ignore added variables.

        Args:
            func: Callable for reward/termination function
        """

        def env_fn(state, action, next_state):
            size = state.size(-1) // 2
            return func(state[..., :-size], action, next_state[..., :-size])

        return env_fn
