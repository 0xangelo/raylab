# pylint:disable=missing-module-docstring
from typing import Any

import gym
from gym.spaces import Box


def assert_box_observation_space(env: gym.Env, obj: Any):
    """Check that `env` has a Box observation space.

    Args:
        env: Gym environment instance
        obj: Object associated with the environment

    Raises:
        AssertionError: If the environment does not have a Box observation
            space
    """
    assert isinstance(env.observation_space, Box), (
        f"{type(obj).__name__} only compatible with Box observation space."
        f" Got: {type(env.observation_space).__name__}."
    )
