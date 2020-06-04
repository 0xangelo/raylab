# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
import math
from typing import Dict
from typing import List

import torch
import torch.nn as nn
from ray.rllib.utils import override

from .abstract import ConditionalDistribution


class Normal(ConditionalDistribution):
    """
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    """

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        loc, scale = torch.chunk(inputs, 2, dim=-1)
        return {"loc": loc, "scale": scale}

    @override(ConditionalDistribution)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        out = self._gen_sample(params, sample_shape).detach()
        return out, self.log_prob(out, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        out = self._gen_sample(params, sample_shape)
        return out, self.log_prob(out, params)

    def _gen_sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int]):
        loc, scale = self._unpack_params(params)
        shape = sample_shape + loc.shape
        eps = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        return loc + eps * scale

    @override(ConditionalDistribution)
    @torch.jit.export
    def log_prob(self, value, params: Dict[str, torch.Tensor]):
        loc, scale = self._unpack_params(params)
        var = scale ** 2
        const = math.log(math.sqrt(2 * math.pi))
        return -((value - loc) ** 2) / (2 * var) - scale.log() - const

    @override(ConditionalDistribution)
    @torch.jit.export
    def cdf(self, value, params: Dict[str, torch.Tensor]):
        loc, scale = self._unpack_params(params)
        return 0.5 * (1 + torch.erf((value - loc) * scale.reciprocal() / math.sqrt(2)))

    @override(ConditionalDistribution)
    @torch.jit.export
    def icdf(self, value, params: Dict[str, torch.Tensor]):
        loc, scale = self._unpack_params(params)
        return loc + scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    @override(ConditionalDistribution)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        _, scale = self._unpack_params(params)
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)

    @override(ConditionalDistribution)
    @torch.jit.export
    def reproduce(self, value, params: Dict[str, torch.Tensor]):
        loc, scale = self._unpack_params(params)
        eps = (value - loc) / scale
        sample_ = loc + scale * eps.detach()
        return sample_, self.log_prob(sample_, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        loc, _ = self._unpack_params(params)
        sample = loc
        return sample, self.log_prob(sample, params)

    def _unpack_params(self, params: Dict[str, torch.Tensor]):
        # pylint:disable=no-self-use
        return params["loc"], params["scale"]
