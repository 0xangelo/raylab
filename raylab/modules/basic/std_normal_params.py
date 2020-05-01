# pylint:disable=missing-module-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class StdNormalParams(nn.Module):
    """Produces standard Normal parameters and expands on input."""

    def __init__(self, input_dim, event_size):
        super().__init__()
        self.input_dim = input_dim
        self.event_shape = (event_size,)

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        shape = inputs.shape[: -self.input_dim] + self.event_shape
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape)}
