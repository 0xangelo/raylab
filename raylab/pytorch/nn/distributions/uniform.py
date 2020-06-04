# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
from typing import Dict
from typing import List

import torch
import torch.nn as nn
from ray.rllib.utils import override

from .abstract import ConditionalDistribution


class Uniform(ConditionalDistribution):
    """
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.
    """

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        low, high = torch.chunk(inputs, 2, dim=-1)
        return {"low": low, "high": high}

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
        low, high = self._unpack_params(params)
        shape = sample_shape + low.shape
        rand = torch.rand(shape, dtype=low.dtype, device=low.device)
        return low + rand * (high - low)

    @override(ConditionalDistribution)
    @torch.jit.export
    def log_prob(self, value, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        lbound = low.le(value).type_as(low)
        ubound = high.gt(value).type_as(low)
        return torch.log(lbound.mul(ubound)) - torch.log(high - low)

    @override(ConditionalDistribution)
    @torch.jit.export
    def cdf(self, value, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        result = (value - low) / (high - low)
        return result.clamp(min=0, max=1)

    @override(ConditionalDistribution)
    @torch.jit.export
    def icdf(self, value, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        return value * (high - low) + low

    @override(ConditionalDistribution)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        return torch.log(high - low)

    @override(ConditionalDistribution)
    @torch.jit.export
    def reproduce(self, value, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        rand = (value - low) / (high - low)
        sample_ = low + rand.detach() * (high - low)
        return sample_, self.log_prob(sample_, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        low, high = self._unpack_params(params)
        sample = (low + high) / 2
        return sample, self.log_prob(sample, params)

    def _unpack_params(self, params: Dict[str, torch.Tensor]):
        # pylint:disable=no-self-use
        return params["low"], params["high"]
