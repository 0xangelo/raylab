# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
import math
from typing import Dict, List

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .abstract import DistributionModule


class Normal(DistributionModule):
    """
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    """

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        loc, scale = torch.chunk(inputs, 2, dim=-1)
        return {"loc": loc, "scale": scale}

    @override(DistributionModule)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        loc, scale = self._unpack_params(params)
        shape = sample_shape + loc.shape
        eps = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        return loc + eps * scale

    @override(DistributionModule)
    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        loc, scale = self._unpack_params(params)
        var = scale ** 2
        const = math.log(math.sqrt(2 * math.pi))
        return -((value - loc) ** 2) / (2 * var) - scale.log() - const

    @override(DistributionModule)
    @torch.jit.export
    def cdf(self, params: Dict[str, torch.Tensor], value):
        loc, scale = self._unpack_params(params)
        return 0.5 * (1 + torch.erf((value - loc) * scale.reciprocal() / math.sqrt(2)))

    @override(DistributionModule)
    @torch.jit.export
    def icdf(self, params: Dict[str, torch.Tensor], prob):
        loc, scale = self._unpack_params(params)
        return loc + scale * torch.erfinv(2 * prob - 1) * math.sqrt(2)

    @override(DistributionModule)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        _, scale = self._unpack_params(params)
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)

    def _unpack_params(self, params: Dict[str, torch.Tensor]):
        # pylint:disable=no-self-use
        return params["loc"], params["scale"]
