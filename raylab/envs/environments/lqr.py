"""
Linear Quadratic Regulator (LQR):
Please see http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
for notation and more details on LQR.
"""
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LQR(nn.Module):
    """Linear Quadratic Regulator solver and simulator."""

    # pylint:disable=invalid-name,abstract-method,missing-function-docstring
    def __init__(self, F: Tensor, f: Tensor, C: Tensor, c: Tensor):
        super().__init__()
        self.F = F.float().detach()
        self.f = f.float().detach()
        self.C = C.float().detach()
        self.c = c.float().detach()

    @property
    def n_dim(self):
        return self.F.shape[1]

    @property
    def state_size(self):
        return self.F.shape[0]

    @property
    def action_size(self):
        return self.n_dim - self.state_size

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
    def backward(self, T):
        policy, value_fn = [], []

        state_size = self.state_size

        V = self.C[:state_size, :state_size]
        v = self.c[:state_size]
        const = 0.0

        value_fn.append((V, v, const))

        for _ in reversed(range(T)):
            K, k, V, v, const_ = self.single_step(V, v)
            const += const_
            policy.append((K, k))
            value_fn.append((V, v, const))

        # policy = list(reversed(policy))
        # value_fn = list(reversed(value_fn[1:]))

        return policy[::-1], value_fn[1:][::-1]

    @torch.jit.export
    def single_step(self, V, v):
        Q, q = self.compute_Q(V, v)
        K, k = self.compute_K(Q, q)
        new_V, new_v = self.compute_V(Q, q, K, k)
        const = self.compute_const(V, v, k)
        return K, k, new_V, new_v, const

    @torch.jit.export
    def compute_Q(self, V, v):
        F, f, C, c = self.F, self.f, self.C, self.c

        FV = F.T @ V
        Q = C + FV @ F
        q = c + FV @ f + F.T @ v
        return Q, q

    @torch.jit.export
    def compute_K(self, Q, q):
        state_size = self.state_size
        Q_uu = Q[state_size:, state_size:]
        Q_ux = Q[state_size:, :state_size]
        q_u = q[state_size:]

        inv_Q_uu = Q_uu.inverse()

        K = -inv_Q_uu @ Q_ux
        k = -inv_Q_uu @ q_u
        return K, k

    @torch.jit.export
    def compute_V(self, Q, q, K, k):
        state_size = self.state_size
        Q_uu = Q[state_size:, state_size:]
        Q_ux = Q[state_size:, :state_size]
        q_u = q[state_size:]
        Q_xx = Q[:state_size, :state_size]
        Q_xu = Q[:state_size, state_size:]
        q_x = q[:state_size]

        K_Q_uu = K.T @ Q_uu

        V = Q_xx + Q_xu @ K + K.T @ Q_ux + K_Q_uu @ K
        v = q_x + Q_xu @ k + K.T @ q_u + K_Q_uu @ k
        return V, v

    @torch.jit.export
    def compute_const(self, V, v, k):
        # pylint:disable=too-many-locals
        state_size = self.state_size
        F, f, C, c = self.F, self.f, self.C, self.c

        V_f = V @ f
        W = C + F.T @ V @ F
        w = c + F.T @ V_f + F.T @ v
        W_uu = W[state_size:, state_size:]
        w_u = w[state_size:]

        const1 = 1 / 2 * k.T @ W_uu @ k
        const2 = k.T @ w_u
        const3 = 1 / 2 * f.T @ V_f + f.T @ v
        const = const1 + const2 + const3
        return const

    @torch.jit.export
    def forward(self, policy: List[Tuple[Tensor, Tensor]], x0: Tensor):
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

        states, actions, costs = map(torch.stack, (states, actions, costs))
        return states, actions, costs
