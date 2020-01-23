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
        if scale:
            self.scale = nn.Parameter(torch.randn(dim))
        else:
            self.register_buffer("scale", torch.zeros(dim))
        if shift:
            self.loc = nn.Parameter(torch.randn(dim))
        else:
            self.register_buffer("loc", torch.zeros(dim))

    @override(NormalizingFlow)
    def _encode(self, inputs):
        out = inputs * torch.exp(self.scale) + self.loc
        log_det = torch.sum(self.scale, dim=-1)
        return out, log_det

    @override(NormalizingFlow)
    def _decode(self, inputs):
        out = (inputs - self.loc) * torch.exp(-self.scale)
        log_det = torch.sum(-self.scale, dim=-1)
        return out, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = not (
            isinstance(self.scale, nn.Parameter) and isinstance(self.loc, nn.Parameter)
        )

    @override(AffineConstantFlow)
    def _encode(self, inputs):
        # first batch is used for init
        if not self.data_dep_init_done:
            stats = inputs.std(dim=0, keepdim=True).log().neg().detach()
            self.scale.data = torch.where(torch.isnan(stats), self.scale.data, stats)
            self.loc.data = (
                (inputs * self.scale.exp()).mean(dim=0, keepdim=True).neg().detach()
            )
            self.data_dep_init_done = True
        return super()._encode(inputs)
