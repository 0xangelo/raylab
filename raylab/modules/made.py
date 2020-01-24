"""
# Copy pasted from
# https://github.com/karpathy/pytorch-made

Implements a Masked Autoregressive MLP, where carefully constructed
binary masks over weights ensure the autoregressive property.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ray.rllib.utils.annotations import override


class MaskedLinear(nn.Linear):
    """Linear module with a configurable mask on the weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        """Update mask tensor."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    @override(nn.Linear)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return F.linear(inputs, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    MADE: Masked Autoencoder for Distribution Estimation

    nin: integer; number of inputs
    hidden sizes: a list of integers; number of units in hidden layers
    nout: integer; number of outputs, which usually collectively parameterize some
        kind of 1D distribution

        Note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then
        the first nin will be all the means and the second nin will be stds. i.e.
        output dimensions depend on the same input dimensions in "chunks" and
        should be carefully decoded downstream appropriately. The output of
        running the tests for this file makes this a bit more clear with examples.
    num_masks: can be used to train ensemble over orderings/connections
    natural_ordering: force natural ordering of dimensions,
        don't use random permutations
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        sizes = [nin] + hidden_sizes + [nout]
        self.net = [
            layer
            for hin, hout in zip(sizes[:-1], sizes[1:])
            for layer in [MaskedLinear(hin, hout), nn.ReLU()]
        ]
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.masks = {}
        self.update_masks()  # builds the initial self.masks connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        """Generate and set masks for each MaskedLinear layer."""
        if self.masks and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        num = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)  # pylint: disable=no-member
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.masks[-1] = (
            np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        )
        for idx in range(num):
            self.masks[idx] = rng.randint(
                self.masks[idx - 1].min(), self.nin - 1, size=self.hidden_sizes[idx]
            )

        # construct the mask matrices
        masks = [
            self.masks[l - 1][:, None] <= self.masks[l][None, :] for l in range(num)
        ]
        masks.append(self.masks[num - 1][:, None] < self.masks[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            factor = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * factor, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.net(inputs)
