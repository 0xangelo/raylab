"""
Reference:

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017
https://arxiv.org/abs/1705.07057

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)
"""
import torch
from torch import nn
from ..made import MADE


class LeafParam(nn.Module):
    """
    just ignores the input and outputs a parameter tensor, lol
    todo maybe this exists in PyTorch somewhere?
    """

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))


class MLP(nn.Module):
    """
    A simple 4-layer MLP.
    Note that the ReLU is not used because it is not an invertible mapping
    (more precisely, it is not a diffeormorphism).
    """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)

    def forward(self, x):
        return self.net(x)


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """

    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim * 2, nh)
        self.parity = parity

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())  # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class IAF(MAF):
    """
    Reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead,
    where sampling will be fast but density estimation slow
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward, self.backward = self.backward, self.forward


class SlowMAF(nn.Module):
    """
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det
