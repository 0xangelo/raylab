"""Mixins for Gym environment wrappers."""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gym.utils import seeding

from .utils import ignore_rightmost_variables


class RNGMixin:
    """Adds a separate random number generator to an environment wrapper.

    Appends the wrapper's rng seed to the list of seeds returned by
    :meth:`seed`.

    Attributes:
        np_random: A numpy RandomState
    """

    # pylint:disable=missing-function-docstring,too-few-public-methods
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.np_random, _ = seeding.np_random()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seeds = super().seed(seed) or []
        self.np_random, seed_ = seeding.np_random(seed)
        return seeds + [seed_]


class IrrelevantRedundantMixin(ABC):
    """Common irrelevant/redundant observation wrapper interface."""

    @property
    @abstractmethod
    def added_size(self):
        """Number of right-most irrelevant/redundant state dimensions."""

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Concatenate irrelevant/redundant variables to the observation."""
        irrelevant_or_redundant = self._added_vars(observation)
        observation = np.concatenate([observation, irrelevant_or_redundant])
        return observation.astype(self.observation_space.dtype)

    @staticmethod
    def wrap_env_function(func: callable, size: int) -> callable:
        """Wrap base env reward/termination function to ignore added variables.

        Args:
            func: Callable for reward/termination function
            size: Number of irrelevant/redundant variables
        """
        return ignore_rightmost_variables(func, size)

    @abstractmethod
    def _added_vars(self, observation: np.ndarray) -> np.ndarray:
        pass

    def _set_reward_if_possible(self):
        if hasattr(self.env, "reward_fn"):
            self.reward_fn = ignore_rightmost_variables(
                self.env.reward_fn, self.added_size
            )

    def _set_termination_if_possible(self):
        if hasattr(self.env, "termination_fn"):
            self.termination_fn = ignore_rightmost_variables(
                self.env.termination_fn, self.added_size
            )
