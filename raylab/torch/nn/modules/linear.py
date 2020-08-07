"""Customized Linear modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils import override

from raylab.torch.nn.init import initialize_


class MaskedLinear(nn.Linear):
    """Linear module with a configurable mask on the weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    @torch.jit.export
    def set_mask(self, mask):
        """Update mask tensor."""
        self.mask.data.copy_(mask)

    @override(nn.Linear)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return F.linear(inputs, self.mask * self.weight, self.bias)


class NormalizedLinear(nn.Module):
    """Enforces a norm constraint on outputs."""

    __constants__ = {"beta"}

    def __init__(self, *args, beta, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.beta = beta
        self.apply(initialize_("xavier_uniform", activation="tanh"))

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        vec = self.linear(inputs)
        norms = vec.norm(p=1, dim=-1, keepdim=True)
        normalized = vec * self.linear.out_features * self.beta / norms
        return torch.where(
            norms / self.linear.out_features > self.beta, normalized, vec
        )
