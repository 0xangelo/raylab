# pylint:disable=missing-module-docstring
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from torch import Tensor

import raylab.torch.nn as nnx
from raylab.torch.nn.init import initialize_
from raylab.torch.nn.utils import get_activation

from .utils import TensorStandardScaler


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
            obs_size, spec.units, spec.activation, layer_norm=spec.layer_norm,
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
        standard_scaler: Whether to transform the inputs of the NN using a
            standard scaling procedure (subtract mean and divide by stddev). The
            transformation mean and stddev should be fitted during training and
            used for both training and evaluation.
    """

    units: List[int] = field(default_factory=list)
    activation: Optional[str] = None
    delay_action: bool = False
    standard_scaler: bool = False


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

        if self.spec.standard_scaler:
            self.obs_scaler = TensorStandardScaler(obs_size)
            self.act_scaler = TensorStandardScaler(action_size)
        else:
            self.obs_scaler = None
            self.act_scaler = None

    @torch.jit.export
    def fit_scaler(self, obs: Tensor, act: Tensor):
        """Fit each sub-scaler to the inputs."""
        if self.obs_scaler is not None:
            self.obs_scaler.fit(obs)
        if self.act_scaler is not None:
            self.act_scaler.fit(act)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        # pylint:disable=arguments-differ
        if self.obs_scaler is not None:
            obs = self.obs_scaler(obs)
        if self.act_scaler is not None:
            act = self.act_scaler(act)
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
