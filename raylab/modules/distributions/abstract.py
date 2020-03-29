"""Distributions as PyTorch modules compatible with TorchScript."""
from typing import Dict, List
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .utils import _sum_rightmost, _multiply_rightmost


class DistributionModule(nn.Module):
    """Implements Distribution interface as a nn.Module."""

    # pylint:disable=abstract-method

    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Returns a (sample, log_prob)
        pair.
        """
        rsample, log_prob = self.rsample(params, sample_shape)
        return rsample.detach(), log_prob

    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Returns a (rsample, log_prob) pair.
        """
        # pylint:disable=unused-argument,no-self-use
        return None, None

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
        entropy = self.entropy(params)
        if entropy is not None:
            return entropy.exp()
        return None

    @torch.jit.export
    def reproduce(self, params: Dict[str, torch.Tensor], value):
        """Produce a reparametrized sample with the same value as `value`."""


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
        out, base_log_prob = self.base_dist.sample(params, sample_shape)
        return out, _sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims)

    @override(DistributionModule)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        out, base_log_prob = self.base_dist.rsample(params, sample_shape)
        if out is not None:
            return out, _sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims)
        return out, base_log_prob

    @override(DistributionModule)
    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        base_log_prob = self.base_dist.log_prob(params, value)
        return _sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims)

    @override(DistributionModule)
    @torch.jit.export
    def cdf(self, params: Dict[str, torch.Tensor], value):
        base_cdf = self.base_dist.cdf(params, value)
        if base_cdf is not None:
            return _multiply_rightmost(base_cdf, self.reinterpreted_batch_ndims)
        return None

    @override(DistributionModule)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        base_entropy = self.base_dist.entropy(params)
        return _sum_rightmost(base_entropy, self.reinterpreted_batch_ndims)

    @override(DistributionModule)
    @torch.jit.export
    def reproduce(self, params: Dict[str, torch.Tensor], value):
        return self.base_dist.reproduce(params, value)


class TransformedDistribution(DistributionModule):
    """
    Extension of the DistributionModule class, which applies a sequence of Transforms
    to a base distribution.
    """

    # pylint:disable=abstract-method

    def __init__(self, base_dist, transform):
        super().__init__()
        self.base_dist = base_dist
        self.transform = transform

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.base_dist(inputs)

    @override(DistributionModule)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        base_sample, base_log_prob = self.base_dist.sample(params, sample_shape)
        transformed, log_abs_det_jacobian = self.transform(base_sample)
        return transformed, base_log_prob + log_abs_det_jacobian

    @override(DistributionModule)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        base_rsample, base_log_prob = self.base_dist.rsample(params, sample_shape)
        transformed, log_abs_det_jacobian = self.transform(base_rsample)
        return transformed, base_log_prob + log_abs_det_jacobian

    @override(DistributionModule)
    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        latent, log_abs_det_jacobian = self.transform(value, reverse=True)
        base_log_prob = self.base_dist.log_prob(params, latent)
        return base_log_prob + log_abs_det_jacobian

    @override(DistributionModule)
    @torch.jit.export
    def reproduce(self, params: Dict[str, torch.Tensor], value):
        latent, _ = self.transform(value, reverse=True)
        latent_ = self.base_dist.reproduce(params, latent)
        if latent_ is not None:
            value_, _ = self.transform(latent_)
            return value_
        return None
