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

from .types import Affine
from .types import LQR as System
from .types import Quadratic


Policy = Affine
Value = Tuple[Tensor, Tensor, Tensor]


class LQRSolver(nn.Module):
    """Linear Quadratic Regulator solver."""

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use

    def forward(self, LQR: System, T: int):
        # pylint:disable=too-many-arguments,arguments-differ
        policy: List[Policy] = []
        value_fn: List[Value] = []

        Vs = self.final_V(LQR)
        value_fn.append(Vs)

        for _ in range(T):  # Effectively solving backwards through time
            Ks, Vs = self.single_step(LQR, Vs)
            policy.insert(0, Ks)
            value_fn.insert(0, Vs)

        return policy, value_fn[:-1]

    def final_V(self, LQR: System) -> Value:
        _, _, C, c = LQR
        state_size = self.state_size(LQR)

        Vs = (C[:state_size, :state_size], c[:state_size], torch.zeros(()))
        return Vs

    def single_step(self, LQR: System, Vs: Value):
        Qs = self.compute_Q(LQR, Vs)
        Ks = self.compute_K(LQR, Qs)
        Vs = self.compute_V(LQR, Qs, Ks, Vs)
        return Ks, Vs

    def compute_Q(self, LQR: System, Vs: Value):
        F, f, C, c = LQR
        V, v, _ = Vs

        FV = F.T @ V
        Q = C + FV @ F
        q = c + FV @ f + F.T @ v
        return Q, q

    def compute_K(self, LQR: System, Qs: Quadratic):
        state_size = self.state_size(LQR)
        Q, q = Qs
        Q_uu = Q[state_size:, state_size:]
        Q_ux = Q[state_size:, :state_size]
        q_u = q[state_size:]

        inv_Q_uu = Q_uu.inverse()

        K = -inv_Q_uu @ Q_ux
        k = -inv_Q_uu @ q_u
        return K, k

    def compute_V(self, LQR: System, Qs: Quadratic, Ks: Affine, Vs: Value):
        # pylint:disable=too-many-locals
        state_size = self.state_size(LQR)
        Q, q = Qs
        Q_uu = Q[state_size:, state_size:]
        Q_ux = Q[state_size:, :state_size]
        q_u = q[state_size:]
        Q_xx = Q[:state_size, :state_size]
        Q_xu = Q[:state_size, state_size:]
        q_x = q[:state_size]

        K, k = Ks
        K_Q_uu = K.T @ Q_uu

        V = Q_xx + Q_xu @ K + K.T @ Q_ux + K_Q_uu @ K
        v = q_x + Q_xu @ k + K.T @ q_u + K_Q_uu @ k
        const = self.compute_const(LQR, Vs, Ks)
        return V, v, const

    def compute_const(self, LQR: System, Vs: Value, Ks: Affine):
        # pylint:disable=too-many-locals
        W_uu, w_u, V_f = self.compute_W(LQR, Vs)
        _, f, _, _ = LQR
        _, v, const_0 = Vs
        _, k = Ks
        const1 = 1 / 2 * k.T @ W_uu @ k
        const2 = k.T @ w_u
        const3 = 1 / 2 * f.T @ V_f + f.T @ v
        const = const_0 + const1 + const2 + const3
        return const.squeeze()

    def compute_W(self, LQR: System, Vs: Value):
        # pylint:disable=too-many-locals
        state_size = self.state_size(LQR)
        F, f, C, c = LQR

        V, v, _ = Vs
        V_f = V @ f
        W = C + F.T @ V @ F
        w = c + F.T @ V_f + F.T @ v
        W_uu = W[state_size:, state_size:]
        w_u = w[state_size:]

        return W_uu, w_u, V_f

    def state_size(self, LQR: System):
        F, _, _, _ = LQR
        return F.shape[0]
