"""Base classes for distribution modules."""
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.utils import override

from . import flows


class ConditionalDistribution(nn.Module):
    """Implements torch.distribution.Distribution interface as a nn.Module.

    If passed a Distribution, wraps the unconditional distribution to be used with the
    ConditionalDistribution interface.
    """

    # pylint:disable=abstract-method,unused-argument,not-callable

    def __init__(self, *, distribution: nn.Module = None):
        super().__init__()
        self.distribution = distribution

    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Returns a (sample, log_prob)
        pair.
        """
        if self.distribution is not None:
            return self.distribution.sample(sample_shape)
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()

    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Returns a (rsample, log_prob) pair.
        """
        if self.distribution is not None:
            return self.distribution.rsample(sample_shape)
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()

    @torch.jit.export
    def log_prob(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        """
        if self.distribution is not None:
            return self.distribution.log_prob(value)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def cdf(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        """Returns the cumulative density/mass function evaluated at `value`."""
        if self.distribution is not None:
            return self.distribution.cdf(value)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def icdf(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        """Returns the inverse cumulative density/mass function evaluated at `value`."""
        if self.distribution is not None:
            return self.distribution.icdf(value)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        """Returns entropy of distribution."""
        if self.distribution is not None:
            return self.distribution.entropy()
        return torch.tensor(np.nan).float()

    @torch.jit.export
    def perplexity(self, params: Dict[str, torch.Tensor]):
        """Returns perplexity of distribution."""
        return self.entropy(params).exp()

    @torch.jit.export
    def reproduce(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        """Produce a reparametrized sample with the same value as `value`."""
        if self.distribution is not None:
            return self.distribution.reproduce(value)
        return (
            torch.tensor(np.nan).float().expand_as(value),
            torch.tensor(np.nan).float().expand_as(value),
        )

    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        """
        Generates a deterministic sample or batch of samples if the distribution
        parameters are batched. Returns a (rsample, log_prob) pair.
        """
        if self.distribution is not None:
            return self.distribution.deterministic()
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()


class Distribution(nn.Module):
    """Unconditional Distribution.

    If passed a ConditionalDistribution, wraps the unconditional distribution to be used
    with the Distribution interface. `nn.Parameter`s passed as distribution parameters
    will be registered as module parameters, making the distribution learnable.
    Otherwise, parameters will be registered as buffers.
    """

    # pylint:disable=abstract-method,not-callable
    params: Dict[str, torch.Tensor]

    def __init__(
        self, *, cond_dist: nn.Module = None, params: Dict[str, torch.Tensor] = None
    ):
        super().__init__()
        self.cond_dist = cond_dist
        self.params = params or {}
        for name, param in self.params.items():
            if isinstance(param, nn.Parameter):
                self.register_parameter(name, param)
            else:
                self.register_buffer(name, param)

    @torch.jit.export
    def sample(self, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Returns a (sample, log_prob)
        pair.
        """
        if self.cond_dist is not None:
            return self.cond_dist.sample(self.params, sample_shape)
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()

    @torch.jit.export
    def rsample(self, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Returns a (rsample, log_prob) pair.
        """
        if self.cond_dist is not None:
            return self.cond_dist.rsample(self.params, sample_shape)
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()

    @torch.jit.export
    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        """
        if self.cond_dist is not None:
            return self.cond_dist.log_prob(value, self.params)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def cdf(self, value):
        """Returns the cumulative density/mass function evaluated at `value`."""
        if self.cond_dist is not None:
            return self.cond_dist.cdf(value, self.params)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def icdf(self, value):
        """Returns the inverse cumulative density/mass function evaluated at `value`."""
        if self.cond_dist is not None:
            return self.cond_dist.icdf(value, self.params)
        return torch.tensor(np.nan).float().expand_as(value)

    @torch.jit.export
    def entropy(self):
        """Returns entropy of distribution."""
        if self.cond_dist is not None:
            return self.cond_dist.entropy(self.params)
        return torch.tensor(np.nan).float()

    @torch.jit.export
    def perplexity(self):
        """Returns perplexity of distribution."""
        return self.entropy().exp()

    @torch.jit.export
    def reproduce(self, value):
        """Produce a reparametrized sample with the same value as `value`."""
        if self.cond_dist is not None:
            return self.cond_dist.reproduce(value, self.params)
        return (
            torch.tensor(np.nan).float().expand_as(value),
            torch.tensor(np.nan).float().expand_as(value),
        )

    @torch.jit.export
    def deterministic(self):
        """
        Generates a deterministic sample or batch of samples if the distribution
        parameters are batched. Returns a (rsample, log_prob) pair.
        """
        if self.cond_dist is not None:
            return self.cond_dist.deterministic(self.params)
        return torch.tensor(np.nan).float(), torch.tensor(np.nan).float()


class Independent(ConditionalDistribution):
    """Reinterprets some of the batch dims of a distribution as event dims."""

    # pylint:disable=abstract-method

    def __init__(self, base_dist, reinterpreted_batch_ndims):
        super().__init__()
        self.base_dist = base_dist
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.base_dist(inputs)

    @override(ConditionalDistribution)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        out, base_log_prob = self.base_dist.sample(params, sample_shape)
        return (
            out,
            flows.utils.sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims),
        )

    @override(ConditionalDistribution)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        out, base_log_prob = self.base_dist.rsample(params, sample_shape)
        if out is not None:
            return (
                out,
                flows.utils.sum_rightmost(
                    base_log_prob, self.reinterpreted_batch_ndims
                ),
            )
        return out, base_log_prob

    @override(ConditionalDistribution)
    @torch.jit.export
    def log_prob(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        base_log_prob = self.base_dist.log_prob(value, params)
        return flows.utils.sum_rightmost(base_log_prob, self.reinterpreted_batch_ndims)

    @override(ConditionalDistribution)
    @torch.jit.export
    def cdf(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        return self.base_dist.cdf(value, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def icdf(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        return self.base_dist.icdf(value, params)

    @override(ConditionalDistribution)
    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        base_entropy = self.base_dist.entropy(params)
        return flows.utils.sum_rightmost(base_entropy, self.reinterpreted_batch_ndims)

    @override(ConditionalDistribution)
    @torch.jit.export
    def reproduce(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        sample_, log_prob_ = self.base_dist.reproduce(value, params)
        return (
            sample_,
            flows.utils.sum_rightmost(log_prob_, self.reinterpreted_batch_ndims),
        )

    @override(ConditionalDistribution)
    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        sample, log_prob = self.base_dist.deterministic(params)
        return (
            sample,
            flows.utils.sum_rightmost(log_prob, self.reinterpreted_batch_ndims),
        )


class TransformedDistribution(ConditionalDistribution):
    """
    Extension of the ConditionalDistribution class, which applies a sequence of
    transformations to a base distribution.
    """

    # pylint:disable=abstract-method

    def __init__(self, base_dist, transform):
        super().__init__()
        self.base_dist = (
            ConditionalDistribution(distribution=base_dist)
            if isinstance(base_dist, Distribution)
            else base_dist
        )
        self.transform = (
            flows.ConditionalTransform(transform=transform)
            if isinstance(transform, flows.Transform)
            else transform
        )

    @override(ConditionalDistribution)
    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        base_sample, base_log_prob = self.base_dist.sample(params, sample_shape)
        transformed, log_abs_det_jacobian = self.transform(base_sample, params)
        return transformed.detach(), base_log_prob - log_abs_det_jacobian

    @override(ConditionalDistribution)
    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        base_rsample, base_log_prob = self.base_dist.rsample(params, sample_shape)
        transformed, log_abs_det_jacobian = self.transform(base_rsample, params)
        return transformed, base_log_prob - log_abs_det_jacobian

    @override(ConditionalDistribution)
    @torch.jit.export
    def log_prob(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        latent, log_abs_det_jacobian = self.transform(value, params, reverse=True)
        base_log_prob = self.base_dist.log_prob(latent, params)
        return base_log_prob + log_abs_det_jacobian

    @override(ConditionalDistribution)
    @torch.jit.export
    def reproduce(self, value: torch.Tensor, params: Dict[str, torch.Tensor]):
        latent, _ = self.transform(value, params, reverse=True)
        latent_, base_log_prob_ = self.base_dist.reproduce(latent, params)
        value_, log_abs_det_jacobian_ = self.transform(latent_, params)
        return value_, base_log_prob_ - log_abs_det_jacobian_

    @override(ConditionalDistribution)
    @torch.jit.export
    def deterministic(self, params: Dict[str, torch.Tensor]):
        base_sample, base_log_prob = self.base_dist.deterministic(params)
        transformed, log_abs_det_jacobian = self.transform(base_sample, params)
        return transformed, base_log_prob - log_abs_det_jacobian
