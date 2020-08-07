"""A simple family of diffeomorphisms."""
import torch
from ray.rllib.utils import override
from torch import nn

from .abstract import Transform


class AffineConstantFlow(Transform):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of
    this where t is None
    """

    def __init__(self, shape, scale=True, shift=True, **kwargs):
        super().__init__(**kwargs)
        if scale:
            self.scale = nn.Parameter(torch.randn(shape))
        else:
            self.register_buffer("scale", torch.zeros(shape))
        if shift:
            self.loc = nn.Parameter(torch.randn(shape))
        else:
            self.register_buffer("loc", torch.zeros(shape))

    @override(Transform)
    def encode(self, inputs):
        out = inputs * torch.exp(self.scale) + self.loc
        # log |dy/dx| = log |torch.exp(scale)| = scale
        log_abs_det_jacobian = self.scale
        return out, log_abs_det_jacobian

    @override(Transform)
    def decode(self, inputs):
        out = (inputs - self.loc) * torch.exp(-self.scale)
        # log |dx/dy| = - log |dy/dx| = - scale
        log_abs_det_jacobian = -self.scale
        return out, log_abs_det_jacobian


class ActNorm(Transform):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, affine_const):
        super().__init__(event_dim=0)
        self.affine_const = affine_const
        self.data_dep_init_done = not (
            isinstance(self.affine_const.scale, nn.Parameter)
            and isinstance(self.affine_const.loc, nn.Parameter)
        )

    @override(Transform)
    def encode(self, inputs):
        # first batch is used for init
        if not self.data_dep_init_done:
            scale = self.affine_const.scale
            loc = self.affine_const.loc

            # pylint:disable=unnecessary-comprehension
            dims = [i for i in range(inputs.dim() - self.event_dim)]
            # pylint:enable=unnecessary-comprehension
            std = -inputs.std(dim=dims).log().detach()
            scale.data.copy_(torch.where(torch.isnan(std), scale, std))
            mean = -torch.mean(inputs * scale.exp(), dim=dims).detach()
            loc.data.copy_(mean)

            self.data_dep_init_done = True
        return self.affine_const.encode(inputs)

    @override(Transform)
    def decode(self, inputs):
        return self.affine_const.decode(inputs)
