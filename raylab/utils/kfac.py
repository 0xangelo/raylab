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
                self._fwd_handles += [mod.register_forward_pre_hook(self.save_input)]
                self._bwd_handles += [mod.register_backward_hook(self.save_grad_out)]
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

    def save_input(self, mod, inputs):
        """Saves input of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]["x"] = inputs[0].detach()

    def save_grad_out(self, mod, _, grad_outputs):
        """Saves grad on output of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]["gy"] = (
                grad_outputs[0] * grad_outputs[0].size(0)
            ).detach()

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
            state.update(self._compute_covs(group, state))
            if state["step"] % self.update_freq == 0:
                state.update(self._process_covs(state))
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
            x = x.data.permute(1, 0, 2).reshape(x.shape[1], -1)
        else:
            x = x.data.T
        batch_size = x.shape[1]

        if len(group["params"]) == 2:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        # Computation of xxt
        xxt = x @ x.T / batch_size
        if "xxt" in state:
            xxt = state["xxt"] * (1.0 - self.alpha) + xxt * self.alpha

        # Computation of ggt
        if group["layer_type"] == "Conv2d":
            gy = gy.data.permute(1, 0, 2, 3)
            num_locations = gy.shape[2] * gy.shape[3]
            gy = gy.reshape(gy.shape[0], -1)
        else:
            gy = gy.data.T
            num_locations = 1

        ggt = gy @ gy.T / batch_size
        if "ggt" in state:
            ggt = state["ggt"] * (1.0 - self.alpha) + ggt * self.alpha

        return {"xxt": xxt, "ggt": ggt, "num_locations": num_locations}

    def _process_covs(self, state):
        """Process the covariances for preconditioning gradients later."""
        xxt, ggt, num_locations = (state[k] for k in "xxt ggt num_locations".split())
        # Computes pi
        pi = 1.0
        if self.pi:
            pi = (torch.trace(xxt) * ggt.shape[0]) / (torch.trace(ggt) * xxt.shape[0])

        # Regularizes and inverts
        eps = self.eps / num_locations
        diag_xxt = torch.diag(torch.empty(xxt.shape[0]).fill_(torch.sqrt(eps * pi)))
        diag_ggt = torch.diag(torch.empty(ggt.shape[0]).fill_(torch.sqrt(eps / pi)))
        ixxt = (xxt + diag_xxt).inverse()
        iggt = (ggt + diag_ggt).inverse()
        return {"ixxt": ixxt, "iggt": iggt}

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group["layer_type"] == "Conv2d" and self.sua:
            return self._precond_sua(weight, bias, state)

        g = weight.grad.data
        if group["layer_type"] == "Conv2d":
            g = g.reshape(g.shape[0], -1)
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)

        ixxt, iggt = state["ixxt"], state["iggt"]
        g = iggt @ g @ ixxt
        if group["layer_type"] == "Conv2d":
            g /= state["num_locations"]

        if bias is not None:
            gb = g[:, -1].reshape_as(bias)
            g = g[:, :-1]
        else:
            gb = None

        g = g.reshape_as(weight)
        return g, gb

    @staticmethod
    def _precond_sua(weight, bias, state):
        """Preconditioning for KFAC SUA."""
        g = weight.grad.data
        s = g.shape
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)

        ixxt, iggt = state["ixxt"], state["iggt"]
        g = ixxt @ g.reshape(-1, s[0] * s[2] * s[3])
        g = g.reshape(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3)
        g = iggt @ g.reshape(s[0], -1)
        g = g.reshape(s[0], -1, s[2], s[3]) / state["num_locations"]
        if bias is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        else:
            gb = None

        return g, gb

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
