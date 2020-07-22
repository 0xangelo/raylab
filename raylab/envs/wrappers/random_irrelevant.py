# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box
from gym.utils.seeding import np_random


class RandomIrrelevant(gym.ObservationWrapper):
    """Add random Normal variables to the environment's observations.

    Args:
        env: Gym environment
        size: Number of random reward-irrelevant variables
        loc: Normal mean
        scale: Normal standard deviation
    """

    def __init__(self, env: gym.Env, size: int, loc: float = 0.0, scale: float = 1.0):
        assert isinstance(
            env.observation_space, Box
        ), f"{type(self).__name__} only compatible with Box observation space"

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

        if not hasattr(self.env, "np_random"):
            self.np_random, _ = np_random()

    def seed(self, seed=None):
        seeds = super().seed(seed) or []
        if self.np_random is not getattr(self.env, "np_random", None):
            self.np_random, seed_ = np_random(seed)
            seeds += [seed_]
        return seeds

    def observation(self, observation: np.ndarray) -> np.ndarray:
        irrelevant = self.np_random.normal(
            loc=self._loc, scale=self._scale, size=self._size
        )
        observation = np.concatenate([observation, irrelevant])
        return observation.astype(self.observation_space.dtype)
