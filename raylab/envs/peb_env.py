"""
Wrapper for Partial Episode Bootstrapping
http://proceedings.mlr.press/v80/pardo18a.html
"""
import gym
from ray.rllib.utils.annotations import override


class IgnoreTimeoutTerminations(gym.Wrapper):
    """Sets 'done = False' if the termination occurs by hitting the time limit."""

    def __init__(self, env=None):
        super().__init__(env)

        _env = env
        while hasattr(_env, "env"):
            if isinstance(_env, gym.wrappers.TimeLimit):
                break
            _env = _env.env
        assert isinstance(
            _env, gym.wrappers.TimeLimit
        ), f"No time limit defined for environment {env}"

    @override(gym.Wrapper)
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = not info.get("TimeLimit.truncated", not done)
        return observation, reward, done, info
