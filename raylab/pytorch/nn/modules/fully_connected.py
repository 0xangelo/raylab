"""Neural network modules using fully connected hidden layers."""
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib.utils import override

from .linear import MaskedLinear
from .utils import get_activation


class FullyConnected(nn.Sequential):
    """Applies several fully connected modules to inputs."""

    def __init__(
        self,
        in_features: int,
        units: Tuple[int, ...] = (),
        activation: str = None,
        layer_norm: bool = False,
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

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint:disable=arguments-differ
        return self.sequential(inputs)


class StateActionEncoder(nn.Module):
    """Concatenates action after the first layer."""

    __constants__ = {"in_features", "out_features"}

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        delay_action: bool = True,
        units: Tuple[int, ...] = (),
        **fc_kwargs
    ):
        super().__init__()
        self.in_features = obs_dim + action_dim
        if units:
            if delay_action is True:
                self.obs_module = FullyConnected(obs_dim, units=units[:1], **fc_kwargs)
                input_dim = units[0] + action_dim
                units = units[1:]
            else:
                self.obs_module = nn.Identity()
                input_dim = obs_dim + action_dim
            self.sequential_module = FullyConnected(input_dim, units=units, **fc_kwargs)
            self.out_features = self.sequential_module.out_features
        else:
            self.obs_module = nn.Identity()
            self.sequential_module = nn.Identity()
            self.out_features = obs_dim + action_dim

    @override(nn.Module)
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # pylint:disable=arguments-differ
        output = self.obs_module(obs)
        output = torch.cat([output, actions], dim=-1)
        output = self.sequential_module(output)
        return output


class MADE(nn.Module):
    """MADE: Masked Autoencoder for Distribution Estimation

    Implements a Masked Autoregressive MLP, where carefully constructed
    binary masks over weights ensure the autoregressive property.

    Based on: https://github.com/karpathy/pytorch-made

    Args:
        in_features: number of inputs
        hidden sizes: number of units in hidden layers
        out_features: number of outputs, which usually collectively parameterize
            some kind of 1D distribution
        natural_ordering: force natural ordering of dimensions,
            don't use random permutations

    Note: if out_features is e.g. 2x larger than `in_features` (perhaps the mean
        and std), then the first `in_features` outputs will be all the means and
        the remaining will be stds. I.e. output dimensions depend on the same
        input dimensions in "chunks" and should be carefully decoded downstream.
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(
        self,
        in_features: int,
        units: Tuple[int, ...],
        out_features: int,
        natural_ordering: bool = False,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert (
            self.out_features % self.in_features == 0
        ), "out_features must be integer multiple of in_features"

        # define a simple MLP neural net
        sizes = [in_features] + list(units) + [out_features]
        # define input units ids
        ids = [torch.arange(sizes[0]) if natural_ordering else torch.randperm(sizes[0])]
        # define hidden units ids
        for idx, size in enumerate(sizes[1:-1]):
            ids.append(torch.randint(ids[idx].min().item(), out_features - 1, (size,)))
        # copy output units ids from input units ids
        ids.append(torch.cat([ids[0]] * (out_features // in_features), dim=-1))
        # define masks for each layer
        masks = [m.unsqueeze(-1) >= n.unsqueeze(0) for m, n in zip(ids[1:-1], ids[:-2])]
        # last layer has a different connection pattern
        masks.append(ids[-1].unsqueeze(-1) > ids[-2].unsqueeze(0))

        linears = [MaskedLinear(hin, hout) for hin, hout in zip(sizes[:-1], sizes[1:])]
        for linear, mask in zip(linears, masks):
            linear.set_mask(mask)
        layers = [m for layer in linears[:-1] for m in (layer, nn.LeakyReLU(0.2))]
        layers += linears[-1:]
        self.net = nn.Sequential(*layers)

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.net(inputs)
