"""Distributions as PyTorch modules compatible with TorchScript."""
from typing import Dict, List
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class DistributionModule(nn.Module):
    """Implements Distribution interface as a nn.Module."""

    # pylint:disable=abstract-method

    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """

    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """

    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        """

    @torch.jit.export
    def cdf(self, params: Dict[str, torch.Tensor], value):
        """Returns the cumulative density/mass function evaluated at `value`."""

    @torch.jit.export
    def icdf(self, params: Dict[str, torch.Tensor], prob):
        """Returns the inverse cumulative density/mass function evaluated at `value`."""

    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        """Returns entropy of distribution."""

    @torch.jit.export
    def perplexity(self, params: Dict[str, torch.Tensor]):
        """Returns perplexity of distribution."""
        return self.entropy(params).exp()


class Independent(DistributionModule):
    """Reinterprets some of the batch dims of a distribution as event dims."""

    # pylint:disable=abstract-method

    def __init__(self, base_dist, reinterpreted_batch_ndims):
        super().__init__()
        self.base_dist = base_dist
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.base_dist(inputs)

    @override(DistributionModule)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        return self.base_dist.sample(params, sample_shape)

    @override(DistributionModule)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        return self.base_dist.rsample(params, sample_shape)

    @override(DistributionModule)
    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        base_log_prob = self.base_dist.log_prob(params, value)
        return _sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims)

    @override(DistributionModule)
    @torch.jit.export
    def cdf(self, params: Dict[str, torch.Tensor], value):
        base_cdf = self.base_dist.cdf(params, value)
        return _multiply_rightmost(base_cdf, self.reinterpreted_batch_ndims)

    @override(DistributionModule)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        base_entropy = self.base_dist.entropy(params)
        return _sum_rightmost(base_entropy, self.reinterpreted_batch_ndims)


def _sum_rightmost(value, dim: int):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def _multiply_rightmost(value, dim: int):
    r"""
    Multiply out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to multiply out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).prod(-1)
