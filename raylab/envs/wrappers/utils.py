# pylint:disable=missing-module-docstring
from typing import Any

from gym.spaces import Box
from gym.spaces import Space


def assert_flat_box_space(space: Space, obj: Any):
    """Check that `space` is a 1D Box space.

    Args:
        env: Gym space instance
        obj: Object associated with the space

    Raises:
        AssertionError: If `space` is not a 1D Box space
    """
    assert isinstance(space, Box), (
        f"{type(obj).__name__} only compatible with 1D Box spaces."
        f" Got: {type(space).__name__}."
    )
    assert len(space.shape) == 1, (
        f"{type(obj).__name__} only compatible with 1D Box spaces."
        f" Got Box space with shape {space.shape}"
    )


def ignore_rightmost_variables(func: callable, size: int) -> callable:
    """Wrap base env reward/termination function to ignore added variables.

    Args:
        func: Callable for reward/termination function
        size: Number of irrelevant/redundant variables
    """

    def env_fn(state, action, next_state):
        return func(state[..., :-size], action, next_state[..., :-size])

    return env_fn
