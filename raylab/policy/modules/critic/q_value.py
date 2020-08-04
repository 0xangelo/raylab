"""Parameterized action-value estimators."""
from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor

from raylab.policy.modules.networks.mlp import StateActionMLP


MLPSpec = StateActionMLP.spec_cls


class QValue(ABC, nn.Module):
    """Neural network module emulating a Q value function."""

    # pylint:disable=arguments-differ
    @abstractmethod
    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Main forward pass mapping obs and actions to Q-values.

        Note:
            The output tensor has a last singleton dimension, i.e., for a batch
            of 10 obs-action pairs, the output will have shape (10, 1).
        """


class MLPQValue(QValue):
    """Q-value function with a multilayer perceptron encoder.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Multilayer perceptron specifications

    Attributes:
        encoder: NN module mapping states to 1D features. Must have an
            `out_features` attribute with the size of the output features
    """

    spec_cls = MLPSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: MLPSpec):
        super().__init__()
        self.encoder = StateActionMLP(obs_space, action_space, spec)
        self.value_linear = nn.Linear(self.encoder.out_features, 1)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        features = self.encoder(obs, action)
        return self.value_linear(features)

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Uses `raylab.pytorch.nn.init.initialize_` to create an initializer
        function.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        self.encoder.initialize_parameters(initializer_spec)


class QValueEnsemble(nn.ModuleList, QValue):
    """A static list of Q-value estimators.

    Args:
        q_values: A list of QValue modules
    """

    def __init__(self, q_values):
        cls_name = type(self).__name__
        assert all(
            isinstance(q, QValue) for q in q_values
        ), f"All modules in {cls_name} must be instances of QValue."
        super().__init__(q_values)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Evaluate each Q estimator in the ensemble.

        Args:
            obs: The observation tensor
            action: The action tensor

        Returns:
            A tensor of shape `(*, N)`, where `N` is the ensemble size
        """
        # pylint:disable=arguments-differ
        return self._action_values(obs, action)

    def _action_values(self, obs: Tensor, act: Tensor) -> Tensor:
        return torch.cat([m(obs, act) for m in self], dim=-1)

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize each Q estimator in the ensemble.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        for q_value in self:
            q_value.initialize_parameters(initializer_spec)


class ForkedQValueEnsemble(QValueEnsemble):
    """Ensemble of Q-value estimators with parallelized forward pass."""

    def _action_values(self, obs: Tensor, act: Tensor) -> Tensor:
        futures = [torch.jit.fork(m, obs, act) for m in self]
        return torch.cat([torch.jit.wait(f) for f in futures], dim=-1)
