# pylint:disable=missing-module-docstring
import gym
import numpy as np
from gym.spaces import Box


class SinglePrecision(gym.Wrapper):
    """Ensures environment outputs are single-precision floats.

    Only compatible with continuous state-action environments.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        try:
            assert isinstance(env.observation_space, Box)
            assert isinstance(env.action_space, Box)
        except AssertionError as err:
            msg = f"{type(self).__name__} only compatible with Box obs/act spaces"
            raise ValueError(msg) from err

        self.observation_space = Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
        )
        self.action_space = Box(
            low=env.action_space.low.astype(np.float32),
            high=env.action_space.high.astype(np.float32),
        )

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(self.observation_space.dtype, copy=False)

    def action(self, action: np.ndarray) -> np.ndarray:
        return action.astype(self.action_space.dtype, copy=False)
