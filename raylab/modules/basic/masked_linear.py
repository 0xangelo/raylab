# pylint:disable=missing-docstring
# pylint:enable=missing-docstring

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override


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
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return F.linear(inputs, self.mask * self.weight, self.bias)
