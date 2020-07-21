# pylint:disable=missing-module-docstring
from abc import ABC
from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor

from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.networks.mlp import StateMLP

from .q_value import QValue


MLPSpec = StateMLP.spec_cls


class VValue(ABC, nn.Module):
    """Neural network module emulating a V value function."""

    # pylint:disable=arguments-differ
    @abstractmethod
    def forward(self, obs: Tensor) -> Tensor:
        """Main forward pass mapping obs to V-values.

        Note:
            The output tensor has a last singleton dimension, i.e., for a batch
            of 10 observations, the output will have shape (10, 1).
        """


class PolicyQValue(VValue):
    """State-value function from policy and Q-value function.

    Args:
        policy: Deterministic policy
        q_value: Q-value function
    """

    def __init__(self, policy: DeterministicPolicy, q_value: QValue):
        super().__init__()
        self.policy = policy
        self.q_value = q_value

    def forward(self, obs: Tensor) -> Tensor:
        act = self.policy(obs)
        value = self.q_value(obs, act)
        return value


class MLPVValue(VValue):
    """V-value function with a multilayer perceptron encoder.

    Args:
        obs_space: Observation space
        spec: Multilayer perceptron specifications

    Attributes:
        encoder: NN module mapping observations to logits
        value_linear: Linear module mapping logits to values
    """

    spec_cls = MLPSpec

    def __init__(self, obs_space: Box, spec: MLPSpec):
        super().__init__()

        self.encoder = StateMLP(obs_space, spec)
        self.value_linear = nn.Linear(self.encoder.out_features, 1)

    def forward(self, obs: Tensor) -> Tensor:
        logits = self.encoder(obs)
        return self.value_linear(logits)


class VValueEnsemble(nn.ModuleList, VValue):
    """A static list of V-value estimators.

    Args:
        v_values: A list of VValue modules
    """

    def __init__(self, v_values: List[VValue]):
        cls_name = type(self).__name__
        assert all(
            isinstance(v, VValue) for v in v_values
        ), f"All modules in {cls_name} must be instances of VValue."
        super().__init__(v_values)

    def forward(self, obs: Tensor) -> Tensor:
        """Evaluate each V estimator in the ensemble.

        Args:
            obs: The observation tensor

        Returns:
            A tensor of shape `(*, N)`, where `N` is the ensemble size
        """
        # pylint:disable=arguments-differ
        return self._state_values(obs)

    def _state_values(self, obs: Tensor) -> Tensor:
        return torch.cat([m(obs) for m in self], dim=-1)


class ForkedVValueEnsemble(VValueEnsemble):
    """Ensemble of V-value estimators with parallelized forward pass."""

    def _state_values(self, obs: Tensor) -> Tensor:
        # pylint:disable=protected-access
        futures = [torch.jit._fork(m, obs) for m in self]
        return torch.cat([torch.jit._wait(f) for f in futures], dim=-1)
