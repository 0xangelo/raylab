"""
Linear Quadratic Regulator (LQR):
Please see http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
for notation and more details on LQR.
"""
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .types import Affine
from .types import LQR


class LQRSim(nn.Module):
    """Linear Quadratic Regulator simulator."""

    # pylint:disable=invalid-name,abstract-method,missing-function-docstring
    def __init__(self, system: LQR):
        super().__init__()
        F, f, C, c = system
        self.F = F.float().detach()
        self.f = f.float().detach()
        self.C = C.float().detach()
        self.c = c.float().detach()

    @torch.jit.export
    def transition(self, x, u):
        inputs = torch.cat([x, u])
        return self.F @ inputs + self.f

    @torch.jit.export
    def cost(self, x, u):
        inputs = torch.cat([x, u])
        c1 = 1 / 2 * inputs.T @ self.C @ inputs
        c2 = inputs @ self.c
        return c1 + c2

    @torch.jit.export
    def final_cost(self, x):
        state_size = self.state_size
        C_xx = self.C[:state_size, :state_size]
        c_x = self.c[:state_size]
        c1 = 1 / 2 * x.T @ C_xx @ x
        c2 = x @ c_x
        return c1 + c2

    @torch.jit.export
    def forward(self, policy: List[Affine], x0: Tensor):
        # pylint:disable=arguments-differ
        states = [x0]
        actions = []
        costs = []

        state = x0

        for K, k in policy:
            action = K @ state + k

            next_state = self.transition(state, action)
            cost = self.cost(state, action)

            state = next_state

            states.append(next_state)
            actions.append(action)
            costs.append(cost)

        final_cost = self.final_cost(state)
        costs.append(final_cost)

        return torch.stack(states), torch.stack(actions), torch.stack(costs)
