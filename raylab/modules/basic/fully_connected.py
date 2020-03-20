# pylint: disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import get_activation, initialize_


class FullyConnected(nn.Sequential):
    """Neural network module that applies several fully connected modules to inputs."""

    def __init__(
        self,
        in_features,
        units=(),
        activation=None,
        layer_norm=False,
        **initializer_options
    ):
        super().__init__()
        self.in_features = in_features
        activ = get_activation(activation)
        units = (self.in_features,) + tuple(units)
        modules = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                modules.append(nn.LayerNorm(out_dim))
            if activ:
                modules.append(activ())
        self.out_features = units[-1]
        self.sequential = nn.Sequential(*modules)

        if "name" in initializer_options:
            self.apply(initialize_(activation=activation, **initializer_options))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.sequential(inputs)
