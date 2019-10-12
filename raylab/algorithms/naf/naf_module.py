"""Normalized Advantage Function neural network modules."""
import torch.nn as nn
from ray.rllib.utils.annotations import override

# pylint: disable=unused-import
from raylab.modules import FullyConnected, TrilMatrix, ActionOutput, ValueFunction

# pylint: enable=unused-import


class NAF(nn.Module):
    """Neural network module implementing the Normalized Advantage Function (NAF)."""

    def __init__(self, logits_module, value_module, advantage_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module
        self.advantage_module = advantage_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        best_value = self.value_module(logits)
        advantage = self.advantage_module(logits, actions)
        return advantage + best_value


class MultivariateGaussianPolicy(nn.Module):
    """Neural network module implementing a multivariate gaussian policy."""

    def __init__(self, logits_module, action_module, tril_module):
        super().__init__()
        self.logits_module = logits_module
        self.action_module = action_module
        self.tril_module = tril_module

    @override(nn.Module)
    def forward(self, obs):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        loc = self.action_module(logits)
        scale_tril = self.tril_module(logits)
        return loc, scale_tril


class AdvantageFunction(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    def __init__(self, tril_module, action_module):
        super().__init__()
        self.tril_module = tril_module
        self.action_module = action_module

    @override(nn.Module)
    def forward(self, logits, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_module(logits)  # square matrix [..., N, N]
        best_action = self.action_module(logits)  # batch of actions [..., N]
        action_diff = (actions - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -0.5 * vec.transpose(-1, -2).matmul(vec).squeeze(-1)
        return advantage
