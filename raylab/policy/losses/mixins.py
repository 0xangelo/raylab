"""Mixins for loss functions."""
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import numpy as np
import torch.nn as nn

from raylab.utils.types import RewardFn
from raylab.utils.types import TerminationFn


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

    def check_env_fns(self):
        """Assert env functions have been set.

        Raises:
            AssertionError: If reward or termination functions were not set
        """
        assert self._env.initialized, "Reward or termination functions missing."


class UniformModelPriorMixin:
    """Add methods for using model ensembles with uniform prior distribution.

    Expects a model ensemble as a `models` instance attribute.

    Attributes:
        models: Module list of dynamics models
    """

    models: nn.ModuleList

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Seeds the RNG for choosing a model from the ensemble."""
        self._rng = np.random.default_rng(seed)

    def sample_model(self) -> Tuple[nn.Module, int]:
        """Return a model and its index sampled uniformly at random."""
        models = self.models
        idx = self._rng.integers(len(models))
        return models[idx], idx
