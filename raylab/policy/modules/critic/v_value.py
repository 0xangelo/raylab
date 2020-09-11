# pylint:disable=missing-module-docstring
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Union

import torch
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor
from torch.jit import fork
from torch.jit import wait

from raylab.policy.modules.actor import Alpha
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.actor import StochasticPolicy
from raylab.policy.modules.networks.mlp import StateMLP

from .q_value import ClippedQValue
from .q_value import QValue
from .q_value import QValueEnsemble


MLPSpec = StateMLP.spec_cls


class VValue(ABC, nn.Module):
    """Neural network module emulating a V value function."""

    # pylint:disable=arguments-differ
    @abstractmethod
    def forward(self, obs: Tensor) -> Tensor:
        """Main forward pass mapping obs to V-values.

        Note:
            Output tensor has scalars for each batch dimension, i.e., for a
            batch of 10 observations, the output will have shape (10,).
        """


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
        return self.value_linear(logits).squeeze(dim=-1)


class VValueEnsemble(nn.ModuleList):
    """A static list of V-value estimators.

    Args:
        v_values: A list of VValue modules
    """

    # pylint:disable=abstract-method
    def __init__(self, v_values: List[VValue]):
        cls_name = type(self).__name__
        assert all(
            isinstance(v, VValue) for v in v_values
        ), f"All modules in {cls_name} must be instances of VValue."
        super().__init__(v_values)

    def forward(self, obs: Tensor) -> List[Tensor]:
        """Evaluate each V estimator in the ensemble.

        Args:
            obs: The observation tensor

        Returns:
            List of `N` output tensors, where `N` is the ensemble size
        """
        # pylint:disable=arguments-differ
        return self._state_values(obs)

    def _state_values(self, obs: Tensor) -> List[Tensor]:
        return [m(obs) for m in self]

    @staticmethod
    def clipped(outputs: List[Tensor]) -> Tensor:
        """Returns the minimum V-value of an ensemble's outputs."""
        mininum, _ = torch.stack(outputs, dim=0).min(dim=0)
        return mininum


class ForkedVValueEnsemble(VValueEnsemble):
    """Ensemble of V-value estimators with parallelized forward pass."""

    # pylint:disable=abstract-method

    def _state_values(self, obs: Tensor) -> List[Tensor]:
        futures = [fork(m, obs) for m in self]
        return [wait(f) for f in futures]


class SoftValue(VValue):
    """V-value computed from stochastic policy, Q-value, and entropy bonus."""

    def __init__(
        self,
        policy: StochasticPolicy,
        q_value: Union[QValue, QValueEnsemble],
        alpha: Alpha,
        deterministic: bool = False,
    ):
        super().__init__()
        if isinstance(q_value, QValueEnsemble):
            # Treat everything as if single value
            q_value = ClippedQValue(q_value)
        self.q_value = q_value

        self.policy = policy
        self.alpha = alpha

        self.deterministic = deterministic

    def forward(self, obs: Tensor) -> Tensor:
        if self.deterministic:
            action, logp = self.policy.deterministic(obs)
        else:
            action, logp = self.policy.rsample(obs)

        alpha = self.alpha()
        entropy_bonus = -alpha * logp
        action_value = self.q_value(obs, action)
        return action_value + entropy_bonus


class HardValue(VValue):
    """V-value computed from deterministic policy and Q-value."""

    def __init__(
        self, policy: DeterministicPolicy, q_value: Union[QValue, QValueEnsemble]
    ):
        super().__init__()
        self.policy = policy

        if isinstance(q_value, QValueEnsemble):
            # Treat everything as if single value
            q_value = ClippedQValue(q_value)
        self.q_value = q_value

    def forward(self, obs):
        return self.q_value(obs, self.policy(obs))


class ClippedVValue(VValue):
    """Minimum of an ensemble of state-value functions."""

    def __init__(self, v_values: VValueEnsemble):
        super().__init__()
        self.v_values = v_values

    def forward(self, obs: Tensor) -> Tensor:
        values = self.v_values(obs)
        minimum, _ = torch.stack(values, dim=0).min(dim=0)
        return minimum
