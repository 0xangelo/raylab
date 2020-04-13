"""
Reference:

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)
"""
import torch
import torch.nn as nn
from .abstract import Transform
from ..made import MADE
from ..basic import LeafParameter


class MLP(nn.Module):
    """A simple 4-layer MLP.

    Note that the ReLU is not used because it is not an invertible mapping
    (more precisely, it is not a diffeormorphism).
    """

    def __init__(self, in_features, out_features, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.net(inputs)


class ARMLP(nn.Module):
    """A 4-layer auto-regressive MLP, wrapper around MADE net."""

    def __init__(self, in_features, out_features, hidden_dim):
        super().__init__()
        self.net = MADE(
            in_features, (hidden_dim,) * 3, out_features, natural_ordering=True,
        )

    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.net(inputs)


class MAF(Transform):
    """Masked Autoregressive Flow that uses a MADE-style network for fast forward."""

    def __init__(self, size, parity, net_class=ARMLP, hidden_size=24, **kwargs):
        # prevent further reduction in superclass
        super().__init__(event_dim=0, **kwargs)
        self.size = size
        self.net = net_class(size, size * 2, hidden_size)
        self.parity = parity

    def encode(self, inputs):
        # we have to decode the x one at a time, sequentially
        out = torch.empty_like(inputs)
        log_abs_det_jacobian = torch.zeros(inputs.shape[:-1])
        inputs = inputs.flip(-1) if self.parity else inputs
        for idx in range(self.size):
            # clone to avoid in-place op errors if using IAF
            scale_shift = self.net(out.clone())
            scale, shift = scale_shift.split(self.size, dim=-1)
            out[..., idx] = (inputs[..., idx] - shift[..., idx]) * torch.exp(
                -scale[..., idx]
            )

            log_abs_det_jacobian += -scale[..., idx]
        return out, log_abs_det_jacobian

    def decode(self, inputs):
        # here we see that we are evaluating all of out in parallel,
        # so density estimation will be fast
        scale_shift = self.net(inputs)
        scale, shift = scale_shift.split(self.size, dim=-1)
        out = inputs * torch.exp(scale) + shift
        # reverse order, so if we stack MAFs correct things happen
        out = out.flip(-1) if self.parity else out

        log_abs_det_jacobian = torch.sum(scale, dim=-1)
        return out, log_abs_det_jacobian


class IAF(MAF):
    """
    Reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead,
    where sampling will be fast but density estimation slow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode, self.decode = self.decode, self.encode


class SlowMAF(Transform):
    """
    Masked Autoregressive Flow; slow version with explicit networks per input feature.
    """

    def __init__(self, size, parity, net_class=MLP, hidden_size=24):
        super().__init__()
        self.size = size
        self.layers = nn.ModuleList()
        self.layers.append(LeafParameter(2))
        for idx in range(1, size):
            self.layers.append(net_class(idx, 2, hidden_size))
        self.order = list(range(size)) if parity else list(range(size))[::-1]

    def encode(self, inputs):
        out = torch.zeros_like(inputs)
        log_abs_det_jacobian = torch.zeros(inputs.size(0))
        for idx, layer in enumerate(self.layers):
            scale_shift = layer(inputs[..., :idx])
            scale, shift = scale_shift[..., 0], scale_shift[..., 1]
            out[..., self.order[idx]] = inputs[..., idx] * torch.exp(scale) + shift
            log_abs_det_jacobian += scale
        return out, log_abs_det_jacobian

    def decode(self, inputs):
        out = torch.zeros_like(inputs)
        log_abs_det_jacobian = torch.zeros(inputs.size(0))
        for idx, layer in enumerate(self.layers):
            scale_shift = layer(out[..., :idx])
            scale, shift = scale_shift[..., 0], scale_shift[..., 1]
            out[..., idx] = (inputs[..., self.order[idx]] - shift) * torch.exp(-scale)
            log_abs_det_jacobian += -scale
        return out, log_abs_det_jacobian
