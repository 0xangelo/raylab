"""
Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.

Based on: https://github.com/karpathy/pytorch-made
"""

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .. import MaskedLinear


class MADE(nn.Module):
    """
    MADE: Masked Autoencoder for Distribution Estimation

    in_features (int): number of inputs
    hidden sizes (List[int]): number of units in hidden layers
    out_features (int): number of outputs, which usually collectively parameterize some
        kind of 1D distribution

        Note: if out_features is e.g. 2x larger than in_features (perhaps the mean and
        std), then the first in_features will be all the means and the second
        in_features will be stds. i.e. output dimensions depend on the same input
        dimensions in "chunks" and should be carefully decoded downstream appropriately.
    natural_ordering (bool): force natural ordering of dimensions,
        don't use random permutations
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, in_features, units, out_features, natural_ordering=False):
        # pylint: disable=too-many-arguments
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
        layers = [m for l in linears[:-1] for m in (l, nn.LeakyReLU(0.2))] + linears[
            -1:
        ]
        self.net = nn.Sequential(*layers)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.net(inputs)
