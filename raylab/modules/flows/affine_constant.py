"""A simple family of diffeomorphisms."""
import torch
from torch import nn
from ray.rllib.utils.annotations import override

from .abstract import NormalizingFlow


class AffineConstantFlow(NormalizingFlow):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of
    this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(dim)) if scale else None
        self.loc = nn.Parameter(torch.randn(dim)) if shift else None

    @override(NormalizingFlow)
    def _encode(self, inputs):
        scale = self.scale or torch.zeros_like(inputs)
        loc = self.loc or torch.zeros_like(inputs)
        out = inputs * torch.exp(scale) + loc
        log_det = torch.sum(scale, dim=-1)
        return out, log_det

    @override(NormalizingFlow)
    def _decode(self, inputs):
        scale = self.scale or torch.zeros_like(inputs)
        loc = self.loc or torch.zeros_like(inputs)
        out = (inputs - loc) * torch.exp(-scale)
        log_det = torch.sum(-scale, dim=1)
        return out, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    @override(AffineConstantFlow)
    def _encode(self, inputs):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.scale is not None and self.loc is not None  # for now
            self.scale.data = torch.log(inputs.std(dim=0, keepdim=True)).neg().detach()
            self.loc.data = (
                (inputs * self.scale.exp()).mean(dim=0, keepdim=True).neg().detach()
            )
            self.data_dep_init_done = True
        return super()._encode(inputs)
