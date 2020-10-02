"""Random LQR problem generators."""
# pylint:disable=invalid-name
from typing import Tuple

import numpy as np
import torch
from numpy.random import Generator
from sklearn.datasets import make_spd_matrix

from .types import Affine
from .types import Box
from .types import LQR
from .types import Quadratic


def box_ddp_random_lqr(
    timestep: float, ctrl_coeff: float, np_random: Generator
) -> Tuple[LQR, Box]:
    # pylint:disable=line-too-long
    """Generate a random, control-limited LQR as described in the Box-DDP paper.

    Taken from `Control-limited differential dynamic programming`_.

    .. _`Control-limited differential dynamic programming`: https://doi.org/10.1109/ICRA.2014.6907001
    """
    # pylint:enable=line-too-long
    assert 0 < timestep < 1

    state_size = np_random.integers(10, 100, endpoint=True)
    ctrl_size = np_random.integers(1, state_size // 2, endpoint=True)

    Fs = _generate_Fs(state_size, ctrl_size, timestep)
    Cs = _generate_Cs(state_size, ctrl_size, timestep, ctrl_coeff)
    bounds = map(torch.from_numpy, (s * np.ones_like(ctrl_size) for s in (-1, 1)))
    return Fs + Cs, bounds


def _generate_Fs(state_size: int, ctrl_size: int, timestep: float) -> Affine:
    F_x = torch.eye(state_size) + timestep * torch.randn(state_size, state_size)
    F_u = torch.randn(state_size, ctrl_size)
    F = torch.cat([F_x, F_u], dim=-1)
    f = torch.zeros(state_size)
    return F, f


def _generate_Cs(
    state_size: int, ctrl_size: int, timestep: float, ctrl_coeff: float
) -> Quadratic:
    dim = state_size + ctrl_size
    C = torch.zeros(dim, dim)

    C_xx = torch.eye(state_size, state_size) * timestep
    C_uu = torch.eye(ctrl_size, ctrl_size) * timestep * ctrl_coeff
    C[:state_size, :state_size] = C_xx
    C[state_size:, state_size:] = C_uu

    c = torch.zeros(dim)
    return C, c


def make_lqr(state_size: int, ctrl_size: int, np_random: Generator) -> LQR:
    """

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    n_dim = state_size + ctrl_size

    F = np_random.normal(size=(state_size, n_dim))
    f = np_random.normal(size=(state_size, 1))

    C = make_spd_matrix(n_dim)
    c = np_random.normal(size=(n_dim, 1))

    return (F, f, C, c)


def make_lqr_linear_navigation(
    goal: Tuple[float, float], beta: float
) -> Tuple[LQR, Box]:
    """

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    state_size = ctrl_size = goal.shape[0]

    F = np.concatenate([np.identity(state_size)] * ctrl_size, axis=1).astype("f")
    f = np.zeros((state_size, 1)).astype("f")

    C = np.diag([2.0] * state_size + [2.0 * beta] * ctrl_size).astype("f")
    c = np.concatenate([-2.0 * goal, np.zeros((ctrl_size, 1))], axis=0).astype("f")
    bounds = map(torch.from_numpy, (s * np.ones_like(ctrl_size) for s in (-1, 1)))
    return (F, f, C, c), bounds
