"""Mixins for loss functions."""
from dataclasses import dataclass
from typing import Optional

from raylab.utils.annotations import RewardFn
from raylab.utils.annotations import TerminationFn


@dataclass
class EnvFunctions:
    """Collection of environment emulating functions."""

    reward: Optional[RewardFn] = None
    termination: Optional[TerminationFn] = None

    @property
    def initialized(self):
        """Whether or not all functions are set."""
        return self.reward is not None and self.termination is not None


class EnvFunctionsMixin:
    """Adds private, externally set environment functions.

    The resulting loss function will have an `_env` private attribute. The user
    can use the `set_reward_fn` and `set_termination_fn` methods to set the
    environment functions. When both are set, `_env.initialized` will be True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env = EnvFunctions()

    def set_reward_fn(self, function: RewardFn):
        """Set reward function to provided callable."""
        self._env.reward = function

    def set_termination_fn(self, function: TerminationFn):
        """Set termination function to provided callable."""
        self._env.termination = function
