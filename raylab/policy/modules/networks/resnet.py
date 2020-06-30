"""
MIT License

Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Slightly modified from:
https://github.com/bayesiains/nsf/blob/master/nn/resnet.py
"""
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from raylab.pytorch.nn.utils import get_activation


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        state_features=None,
        activation="ReLU",
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        activation = get_activation(activation)

        layers = []
        if use_batch_norm:
            layers += [nn.BatchNorm1d(features, eps=1e-3)]
        layers += [activation()]
        layers += [nn.Linear(features, features)]

        if use_batch_norm:
            layers += [nn.BatchNorm1d(features, eps=1e-3)]
        layers += [activation()]
        layers += [nn.Dropout(p=dropout_probability)]
        layers += [nn.Linear(features, features)]
        if zero_initialization:
            nn.init.uniform_(layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(layers[-1].bias, -1e-3, 1e-3)

        self.sequential = nn.Sequential(*layers)

        self.state_layer = None
        if state_features is not None:
            self.state_layer = nn.Linear(state_features, features)

    def forward(self, inputs, params: Optional[Dict[str, torch.Tensor]] = None):
        # pylint:disable=arguments-differ
        temps = inputs
        temps = self.sequential(temps)
        if self.state_layer is not None:
            if params is None:
                raise ValueError("Parameters required for stateful block.")
            temps = F.glu(
                torch.cat((temps, self.state_layer(params["state"])), dim=-1), dim=-1
            )
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        state_features=None,
        num_blocks=2,
        activation="ReLU",
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        self.hidden_features = hidden_features
        self.stateful = bool(state_features)
        if self.stateful:
            self.initial_layer = nn.Linear(
                in_features + state_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    state_features=state_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, params: Optional[Dict[str, torch.Tensor]] = None):
        # pylint:disable=arguments-differ
        if self.stateful:
            if params is None:
                raise ValueError("Parameters required for stateful resnet.")
            temps = self.initial_layer(torch.cat((inputs, params["state"]), dim=-1))
        else:
            temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps, params)
        outputs = self.final_layer(temps)
        return outputs
