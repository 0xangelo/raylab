# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_


class NormalizedLinear(nn.Module):
    """Neural network module that enforces a norm constraint on outputs."""

    __constants__ = {"beta"}

    def __init__(self, *args, beta, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.beta = beta
        self.apply(initialize_("xavier_uniform", activation="tanh"))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        vec = self.linear(inputs)
        norms = vec.norm(p=1, dim=-1, keepdim=True)
        normalized = vec * self.linear.out_features * self.beta / norms
        return torch.where(
            norms / self.linear.out_features > self.beta, normalized, vec
        )
