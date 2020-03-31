"""Base classes for Normalizing Flows."""
import torch
from torch import nn
from ray.rllib.utils.annotations import override


class NormalizingFlow(nn.Module):
    """A diffeomorphism."""

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self._encode(inputs) if self.training else self._decode(inputs)

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
    def forward(self, inputs):  # pylint: disable=arguments-differ
        out, log_det = inputs, torch.zeros([])
        for flow in self.flows:
            out, log_det_ = flow(out)
            log_det = log_det + log_det_
        return out, log_det


class NormalizingFlowModel(nn.Module):
    """A Normalizing Flow Model is a (prior, flow) pair."""

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = ComposeNormalizingFlow(flows)

    @override(nn.Linear)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        if self.training:
            out, log_det = self.flow(inputs)
            prior_logp = self.prior.log_prob(out)
            return out, prior_logp + log_det

        out = self.prior.sample((inputs,))
        prior_logp = self.prior.log_prob(out)
        out, log_det = self.flow(out)
        return out, prior_logp + log_det
