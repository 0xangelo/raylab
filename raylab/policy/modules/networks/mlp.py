# pylint:disable=missing-module-docstring
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from torch import Tensor

import raylab.torch.nn as nnx
from raylab.torch.nn.init import initialize_
from raylab.torch.nn.utils import get_activation


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


class StateMLP(nnx.FullyConnected):
    """Multilayer perceptron for encoding state inputs."""

    spec_cls = StateMLPSpec

    def __init__(self, obs_space: Box, spec: StateMLPSpec):
        obs_size = obs_space.shape[0]
        super().__init__(
            obs_size,
            spec.units,
            spec.activation,
            layer_norm=spec.layer_norm,
        )
        self.spec = spec

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Uses `raylab.torch.nn.init.initialize_` to create an initializer
        function.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        initializer = initialize_(activation=self.spec.activation, **initializer_spec)
        self.apply(initializer)


@dataclass
class StateActionMLPSpec(DataClassJsonMixin):
    """Specifications for building an MLP with state and action inputs.

    Args:
        units: Number of units in each hidden layer
        activation: Nonlinearity following each linear layer
        delay_action: Whether to apply an initial preprocessing layer on the
            observation before concatenating the action to the input.
    """

    units: List[int] = field(default_factory=list)
    activation: Optional[str] = None
    delay_action: bool = False


class StateActionMLP(nn.Module):
    """Multilayer perceptron for encoding state-action inputs."""

    spec_cls = StateActionMLPSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: StateActionMLPSpec):
        super().__init__()
        self.spec = spec
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        self.encoder = nnx.StateActionEncoder(
            obs_size,
            action_size,
            units=spec.units,
            activation=spec.activation,
            delay_action=spec.delay_action,
        )
        self.out_features = self.encoder.out_features

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        # pylint:disable=arguments-differ
        return self.encoder(obs, act)

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Uses `raylab.torch.nn.init.initialize_` to create an initializer
        function.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        initializer = initialize_(activation=self.spec.activation, **initializer_spec)
        self.encoder.apply(initializer)


class LayerNormMLP(nn.Sequential):
    # pylint:disable=line-too-long
    """Simple feedforward MLP torso with initial layer-norm.

    This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
    first layer and non-linearities (elu) on all but the last remaining layers.

    Taken from `Acme`_.

    .. _`Acme`: https://github.com/deepmind/acme/blob/ace9f280cb9c6ba954e909bb0340c6b5ac45e4c6/acme/tf/networks/continuous.py#L37

    Args:
        layer_sizes: A sequence of ints specifying the size of each layer.
        activate_final: Whether to use the activation function on the final
            layer of the neural network.
    """
    # pylint:disable=line-too-long

    def __init__(
        self, in_size: int, layer_sizes: Sequence[int], activate_final: bool = False
    ):
        units = list(layer_sizes)
        assert units, f"Must pass at least 1 layer size to {type(self).__name__}"

        mlp = (
            nnx.FullyConnected(
                in_features=layer_sizes[0],
                units=layer_sizes[1:],
                activation="ELU",
                layer_norm=False,
            ),
        )
        if not activate_final:
            mlp = mlp[:-1]

        super().__init__(
            nn.Linear(in_size, layer_sizes[0]),
            nn.LayerNorm(layer_sizes[0]),
            nn.Tanh(),
            mlp,
        )


class MLP(nn.Module):
    """A general purpose Multi-Layer Perceptron."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        state_features=None,
        num_blocks=2,
        activation="ReLU",
        activate_output=False,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        activation = get_activation(activation)
        self.stateful = bool(state_features)
        if self.stateful:
            self.initial_layer = nn.Linear(
                in_features + state_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)

        layers = [activation()]
        layers += [
            layer
            for _ in range(num_blocks)
            for layer in (nn.Linear(hidden_features, hidden_features), activation())
        ]
        layers += [nn.Linear(hidden_features, out_features)]

        if activate_output:
            layers += [activation()]

        self.sequential = nn.Sequential(*layers)

    def forward(self, inputs, params: Optional[Dict[str, torch.Tensor]] = None):
        # pylint:disable=arguments-differ
        if self.stateful:
            if params is None:
                raise ValueError("Parameters required for stateful mlp.")
            out = self.initial_layer(torch.cat([inputs, params["state"]], dim=-1))
        else:
            out = self.initial_layer(inputs)

        return self.sequential(out)
