"""Mixins for loss functions."""
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

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


class UniformModelPriorMixin:
    """Add methods for using model ensembles with uniform prior distribution.

    Expects a model ensemble as a `_models` instance attribute.

    Attributes:
        grad_estimator: Gradient estimator for expecations ('PD' or 'SF')
    """

    grad_estimator: str = "PD"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Seeds the RNG for choosing a model from the ensemble."""
        self._rng = np.random.default_rng(seed)

    def transition(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute virtual transition and its log density.

        Samples a model from the ensemble using the internal RNG and uses it to
        generate the next state.

        Args:
            obs: The current state
            action: The action sampled from the stochastic policy

        Returns:
            A tuple with the next state, its log-likelihood generated from
            a model sampled from the ensemble, and that model's distribution
            parameters
        """
        model = self._rng.choice(self._models)
        dist_params = model.params(obs, action)
        if self.grad_estimator == "SF":
            next_obs, logp = model.dist.sample(dist_params)
        elif self.grad_estimator == "PD":
            next_obs, logp = model.dist.rsample(dist_params)
        return next_obs, logp, dist_params

    def verify_model(self, obs: Tensor, act: Tensor):
        """Verify model suitability for the current gradient estimator.

        Assumes all models in the ensemble behave the same way.

        Args:
            obs: Dummy observation tensor
            act: Dummy action tensor

        Raises:
            AssertionError: If the internal model does not satisfy requirements
                for gradient estimation
        """
        model = self._models[0]
        if self.grad_estimator == "SF":
            sample, logp = model.sample(obs, act.requires_grad_())
            assert sample.grad_fn is None
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
        if self.grad_estimator == "PD":
            sample, _ = model.rsample(obs.requires_grad_(), act.requires_grad_())
            sample.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
