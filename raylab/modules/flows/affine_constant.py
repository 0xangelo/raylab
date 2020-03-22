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


class ActNorm(NormalizingFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.affine_const = AffineConstantFlow(*args, **kwargs)
        self.data_dep_init_done = not (
            isinstance(self.affine_const.scale, nn.Parameter)
            and isinstance(self.affine_const.loc, nn.Parameter)
        )

    @override(NormalizingFlow)
    def _encode(self, inputs):
        # first batch is used for init
        if not self.data_dep_init_done:
            scale = self.affine_const.scale
            loc = self.affine_const.loc

            # pylint:disable=unnecessary-comprehension
            dims = [i for i in range(inputs.dim() - 1)]
            # pylint:enable=unnecessary-comprehension
            std = -inputs.std(dim=dims).log().detach()
            scale.data.copy_(torch.where(torch.isnan(std), scale, std))
            mean = -torch.mean(inputs * scale.exp(), dim=dims).detach()
            loc.data.copy_(mean)

            self.data_dep_init_done = True
        return self.affine_const(inputs)

    @override(NormalizingFlow)
    def _decode(self, inputs):
        return self.affine_const(inputs)
