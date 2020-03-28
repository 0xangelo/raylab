"""Distributions as PyTorch modules compatible with TorchScript."""
from typing import Dict, List
import torch
import torch.nn as nn


class DistributionModule(nn.Module):
    """Implements Distribution interface as a nn.Module."""

    # pylint:disable=abstract-method

    @torch.jit.export
    def cdf(self, params: Dict[str, torch.Tensor], value):
        """Returns the cumulative density/mass function evaluated at `value`."""

    @torch.jit.export
    def entropy(self, params: Dict[str, torch.Tensor]):
        """Returns entropy of distribution."""

    @torch.jit.export
    def icdf(self, params: Dict[str, torch.Tensor], prob):
        """Returns the inverse cumulative density/mass function evaluated at `value`."""

    @torch.jit.export
    def log_prob(self, params: Dict[str, torch.Tensor], value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        """

    @torch.jit.export
    def perplexity(self, params: Dict[str, torch.Tensor]):
        """Returns perplexity of distribution."""

    @torch.jit.export
    def rsample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """

    @torch.jit.export
    def sample(self, params: Dict[str, torch.Tensor], sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
