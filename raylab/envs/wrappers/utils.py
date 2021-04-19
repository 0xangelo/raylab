# pylint:disable=missing-module-docstring
import textwrap
from typing import Any

from gym.spaces import Box, Space


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


def check_redundant_size_compat(size: int, space: Box):
    """Check that `size` is less or equal to the size of the observations."""
    assert size <= space.shape[0], textwrap.dedent(
        f"""\
        Can only compute redundant variables from a subset of observation features.
        `size` must be less or equal to the size of the base env observations.
        Got '{size}', whereas the observation shape is {space.shape}
        """
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
