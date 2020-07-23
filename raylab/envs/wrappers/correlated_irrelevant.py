# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .mixins import RNGMixin
from .utils import assert_box_observation_space


class CorrelatedIrrelevant(RNGMixin, gym.ObservationWrapper):
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
