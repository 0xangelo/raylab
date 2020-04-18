# pylint:disable=missing-module-docstring
from typing import Dict, Optional

import torch
import torch.nn as nn

import raylab.utils.pytorch as ptu


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
        activation = ptu.get_activation(activation)
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
