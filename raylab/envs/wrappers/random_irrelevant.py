# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import RNGMixin
from .utils import assert_box_observation_space


class RandomIrrelevant(RNGMixin, gym.ObservationWrapper):
    """Add Normal random variables to the environment's observations.

    Args:
        env: Gym environment
        size: Number of random reward-irrelevant variables
        loc: Normal mean
        scale: Normal standard deviation
    """

    def __init__(self, env: gym.Env, size: int, loc: float = 0.0, scale: float = 1.0):
        assert_box_observation_space(env, self)
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

    def observation(self, observation: np.ndarray) -> np.ndarray:
        irrelevant = self.np_random.normal(
            loc=self._loc, scale=self._scale, size=self._size
        )
        observation = np.concatenate([observation, irrelevant])
        return observation.astype(self.observation_space.dtype)

    @staticmethod
    def wrap_env_function(func: callable, size: int) -> callable:
        """Wrap base env reward/termination function to ignore added variables.

        Args:
            func: Callable for reward/termination function
            size: Number of irrelevant/redundant variables
        """

        def env_fn(state, action, next_state):
            return func(state[..., :-size], action, next_state[..., :-size])

        return env_fn
