# pylint:disable=missing-module-docstring
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from torch import Tensor

import raylab.pytorch.nn as nnx


@dataclass
class StateMLPSpec(DataClassJsonMixin):
    """Specifications for creating a multilayer perceptron.

    Args:
    units: Number of units in each hidden layer
    activation: Nonlinearity following each linear layer
    layer_norm: Whether to apply layer normalization between each linear layer
        and following activation
    """

    units: List[int] = field(default_factory=list)
    activation: Optional[str] = None
    layer_norm: bool = False


class StateMLP(nn.Module):
    """Multilayer perceptron for encoding state inputs.

    Attributes:
        encoder: Fully connected module with multiple layers
    """

    spec_cls = StateMLPSpec

    def __init__(self, obs_space: Box, spec: StateMLPSpec):
        super().__init__()
        obs_size = obs_space.shape[0]
        self.encoder = nnx.FullyConnected(
            obs_size, spec.units, spec.activation, layer_norm=spec.layer_norm,
        )

    def forward(self, obs: Tensor) -> Tensor:
        # pylint:disable=arguments-differ
        return self.encoder(obs)
