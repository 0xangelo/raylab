"""Base classes for Normalizing Flows."""
import torch
from torch import nn
from ray.rllib.utils.annotations import override


class NormalizingFlow(nn.Module):
    """A diffeomorphism.

    All flows map data to a latent space by default (f(x) -> z).
    Use the `reverse` flag to invert the flow (f^{-1}(z) -> x).
    """

    @override(nn.Module)
    def forward(self, inputs, reverse: bool = False):
        # pylint: disable=arguments-differ
        return self._decode(inputs) if reverse else self._encode(inputs)

    def _encode(self, inputs):
        """
        Apply the forward transformation to the data.

        Maps data points to latent variables, returning the transformed variable and the
        log of the absolute Jacobian determinant.
        """

    def _decode(self, inputs):
        """
        Apply the inverse transformation to the data.

        Maps latent variables to datapoints, returning the transformed variable and the
        log of the absolute Jacobian determinant.
        """


class ComposeNormalizingFlow(nn.Module):
    """A composition of Normalizing Flows is a Normalizing Flow."""

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    @override(nn.Linear)
    def forward(self, inputs, reverse: bool = False):
        # pylint: disable=arguments-differ
        out, log_det = inputs, torch.zeros([])
        for flow in self.flows:
            out, log_det_ = flow(out, reverse=reverse)
            log_det = log_det + log_det_
        return out, log_det


class NormalizingFlowModel(nn.Module):
    """A (prior, flow) pair that allows density estimation and sampling.

    The forward pass encodes the inputs and returns their log-likelihood.
    Use `rsample` to produce samples.
    """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.prior_rsample = prior.rsample
        self.prior_logp = prior.log_prob
        self.flow = ComposeNormalizingFlow(flows)

    @torch.jit.export
    def rsample(self, n_samples: int = 1):
        """Produce a reparameterized sample."""
        sample = self.prior_rsample((n_samples,))
        prior_logp = self.prior_logp(sample)
        sample, log_det = self.flow(sample, reverse=True)
        return sample, prior_logp + log_det

    @override(nn.Module)
    def forward(self, inputs):
        # pylint: disable=arguments-differ
        latent, log_det = self.flow(inputs)
        prior_logp = self.prior_logp(latent)
        return latent, prior_logp + log_det
