# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box

from .utils import assert_box_observation_space


class NonlinearRedundant(gym.ObservationWrapper):
    r"""Adds nonlinear redundant variables to the environment's observations.

    Computes redundant variables as nonlinear functions.
    .. math::
        (x_t, \cos(x_t), \sin(x_t)) \in \mathbb{R}^{3d}

    Args:
        env: Gym environment instance
    """

    def __init__(self, env: gym.Env):
        assert_box_observation_space(env, self)
        super().__init__(env)

        original = self.env.observation_space
        size = original.shape[0]
        low = np.concatenate([original.low, [-1] * 2 * size]).astype(original.dtype)
        high = np.concatenate([original.high, [1] * 2 * size]).astype(original.dtype)
        self.observation_space = Box(low=low, high=high)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        cos = np.cos(observation)
        sin = np.sin(observation)
        observation = np.concatenate([observation, cos, sin])
        return observation.astype(self.observation_space.dtype)

    @staticmethod
    def wrap_env_function(func: callable) -> callable:
        """Wrap base env reward/termination function to ignore added variables.

        Args:
            func: Callable for reward/termination function
        """

        def env_fn(state, action, next_state):
            size = (state.size(-1) // 3) * 2
            return func(state[..., :-size], action, next_state[..., :-size])

        return env_fn
