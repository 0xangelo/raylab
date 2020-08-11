# pylint:disable=missing-docstring
import torch
import torch.nn as nn
from torch import Tensor


class TanhSquash(nn.Module):
    """Neural network module squashing vectors to specified range using Tanh."""

    def __init__(self, low: Tensor, high: Tensor):
        super().__init__()
        self.register_buffer("loc", (high + low) / 2)
        self.register_buffer("scale", (high - low) / 2)

    def forward(self, inputs: Tensor, reverse: bool = False) -> Tensor:
        # pylint:disable=arguments-differ
        if reverse:
            inputs = (inputs - self.loc) / self.scale
            to_log1 = torch.clamp(1 + inputs, min=1.1754943508222875e-38)
            to_log2 = torch.clamp(1 - inputs, min=1.1754943508222875e-38)
            return (torch.log(to_log1) - torch.log(to_log2)) / 2
        return self.loc + inputs.tanh() * self.scale
