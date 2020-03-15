# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_


class NormalizedLinear(nn.Linear):
    """Neural network module that enforces a norm constraint on outputs."""

    __constants__ = set(nn.Linear.__constants__ + ["beta"])

    def __init__(self, *args, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.apply(initialize_("xavier_uniform", activation="tanh"))

    @override(nn.Linear)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        vec = super().forward(inputs)
        norms = vec.norm(p=1, dim=-1, keepdim=True)
        normalized = vec * self.out_features * self.beta / norms
        return torch.where(norms / self.out_features > self.beta, normalized, vec)
