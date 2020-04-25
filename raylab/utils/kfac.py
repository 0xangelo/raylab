"""
MIT License

Copyright (c) 2018 Thomas George, César Laurent and Université de Montréal.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Adapted from: https://github.com/Thrandis/EKFAC-pytorch
"""
import contextlib
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


class KFACOptimizer(Optimizer):
    """K-FAC Optimizer for Linear and Conv2d layers.

    Computes the K-FAC of the second moment of the gradients.
    It works for Linear and Conv2d layers and silently skip other layers.

    Args:
        net (torch.nn.Module): Network to optimize.
        eps (float): Tikhonov regularization parameter for the inverses.
        sua (bool): Applies SUA approximation.
        pi (bool): Computes pi correction for Tikhonov regularization.
        update_freq (int): Perform inverses every update_freq updates.
        alpha (float): Running average parameter (if == 1, no r. ave.).
        kl_clip (float): Scale the gradients by the squared fisher norm.
        eta (float): upper bound for gradient scaling.
    """

    # pylint:disable=invalid-name,too-many-instance-attributes

    def __init__(
        self,
        net,
        eps,
        sua=False,
        pi=False,
        update_freq=1,
        alpha=1.0,
        kl_clip=1e-3,
        eta=1.0,
        lr=1.0,
    ):
        # pylint:disable=too-many-arguments,too-many-locals
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.eta = eta
        self._fwd_handles = []
        self._bwd_handles = []
        self._recording = False

        param_groups = []
        param_set = set()
        for mod in net.modules():
            mod_class = type(mod).__name__
            if mod_class in ["Linear", "Conv2d"]:
                self._fwd_handles += [mod.register_forward_pre_hook(self._save_input)]
                self._bwd_handles += [mod.register_backward_hook(self._save_grad_out)]
                info = (
                    (mod.kernel_size, mod.padding, mod.stride)
                    if mod_class == "Conv2d"
                    else None
                )
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                param_groups.append(
                    {"params": params, "info": info, "layer_type": mod_class}
                )
                param_set.update(set(params))

        param_groups.append(
            {"params": [p for p in net.parameters() if p not in param_set]}
        )
        super(KFACOptimizer, self).__init__(param_groups, {"lr": lr})
        self.state["kl_clip"] = kl_clip

    def step(self):  # pylint:disable=arguments-differ
        """Preconditions and applies gradients."""
        fisher_norm = 0.0
        for group in self.param_groups[:-1]:
            # Getting parameters
            params = group["params"]
            weight, bias = params if len(params) == 2 else (params[0], None)
            state = self.state[weight]

            # Update convariances and inverses
            state.setdefault("step", 0)
            self._compute_covs(group, state)
            if state["step"] % self.update_freq == 0:
                ixxt, iggt = self._inv_covs(
                    state["xxt"], state["ggt"], state["num_locations"]
                )
                state.update((("ixxt", ixxt), ("iggt", iggt)))
            state["step"] += 1

            # Preconditionning
            gw, gb = self._precond(weight, bias, group, state)
            # Updating gradients
            fisher_norm += (weight.grad * gw).sum()
            weight.grad.data = gw
            if bias is not None:
                fisher_norm += (bias.grad * gb).sum()
                bias.grad.data = gb

            # Cleaning
            self.state[weight].pop("x", None)
            self.state[weight].pop("gy", None)

        fisher_norm += sum(
            (p.grad * p.grad).sum() for p in self.param_groups[-1]["params"]
        )

        # Eventually scale the norm of the gradients and apply each
        scale = min(self.eta, torch.sqrt(self.state["kl_clip"] / fisher_norm))
        for group in self.param_groups:
            for param in group["params"]:
                param.grad.data.mul_(scale)
                param.data.sub_(group["lr"], param.grad.data)

    @contextlib.contextmanager
    def record_stats(self):
        """Activate registered forward and backward hooks."""
        try:
            self._recording = True
            yield
        except Exception as excep:
            raise excep
        finally:
            self._recording = False

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]["x"] = i[0]

    def _save_grad_out(self, mod, _, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]["gy"] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group["layer_type"] == "Conv2d" and self.sua:
            return self._precond_sua(weight, bias, state)

        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        if group["layer_type"] == "Conv2d":
            g = g.contiguous().view(s[0], s[1] * s[2] * s[3])

        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)

        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group["layer_type"] == "Conv2d":
            g /= state["num_locations"]

        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None

        g = g.contiguous().view(*s)
        return g, gb

    @staticmethod
    def _precond_sua(weight, bias, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)

        g = torch.mm(ixxt, g.contiguous().view(-1, s[0] * s[2] * s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state["num_locations"]
        if bias is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        else:
            gb = None

        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        x, gy = state["x"], state["gy"]
        # Computation of xxt
        if group["layer_type"] == "Conv2d":
            if not self.sua:
                kernel_size, padding, stride = group["info"]
                x = F.unfold(x, kernel_size, padding=padding, stride=stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()

        if len(group["params"]) == 2:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        if state["step"] == 0:
            state["xxt"] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state["xxt"].addmm_(
                mat1=x,
                mat2=x.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(x.shape[1]),
            )

        # Computation of ggt
        if group["layer_type"] == "Conv2d":
            gy = gy.data.permute(1, 0, 2, 3)
            state["num_locations"] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state["num_locations"] = 1

        if state["step"] == 0:
            state["ggt"] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state["ggt"].addmm_(
                mat1=gy,
                mat2=gy.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(gy.shape[1]),
            )

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverts the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = tx / tg

        # Regularizes and inverts
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
