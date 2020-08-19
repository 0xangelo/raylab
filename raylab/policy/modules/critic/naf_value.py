"""Parameterized normalized advantage function estimators."""
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor

import raylab.torch.nn as nnx
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy

from .q_value import QValue
from .v_value import VValue


class NAFQValue(QValue):
    """Neural network module emulating a Normalized Advantage Function.

    Args:
        action_space: Continuous action space
        policy: Continuous deterministic policy

    Attributes:
        policy: Deterministic policy
        v_value: State-value function
    """

    def __init__(self, action_space: Box, policy: DeterministicPolicy):
        super().__init__()
        act_size = action_space.shape[0]
        self.policy = policy

        self._value_linear = nn.Linear(policy.encoder.out_features, 1)
        self.v_value = NAFVValue(policy.encoder, self._value_linear)

        self._tril = nnx.TrilMatrix(policy.encoder.out_features, act_size)
        self._advantage = AdvantageFunction(
            self._tril, nn.Sequential(policy.action_linear, policy.squashing)
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        logits = self.policy.encoder(obs)
        best_value = self._value_linear(logits).squeeze(-1)
        advantage = self._advantage(logits, action).squeeze(-1)
        return advantage + best_value


class NAFVValue(VValue):
    """Wrapper around NAF's state-value function."""

    def __init__(self, encoder: nn.Module, value_linear: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.value_linear = value_linear

    def forward(self, obs: Tensor) -> Tensor:
        return self.value_linear(self.encoder(obs)).squeeze(-1)


class AdvantageFunction(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    def __init__(self, tril_module, action_module):
        super().__init__()
        self.tril_module = tril_module
        self.action_module = action_module

    def forward(self, logits, action):  # pylint:disable=arguments-differ
        tril_matrix = self.tril_module(logits)  # square matrix [..., N, N]
        best_action = self.action_module(logits)  # batch of actions [..., N]
        action_diff = (action - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -0.5 * vec.transpose(-1, -2).matmul(vec).squeeze(-1)
        return advantage  # scalars [..., 1]
