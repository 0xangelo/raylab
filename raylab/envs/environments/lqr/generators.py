"""Random LQR problem generators."""
# pylint:disable=invalid-name
from typing import Tuple

import numpy as np
import torch

from .types import Affine
from .types import Box
from .types import LQR
from .types import Quadratic


def box_ddp_random_lqr(timestep: float, ctrl_coeff: float) -> Tuple[LQR, Box]:
    # pylint:disable=line-too-long
    """Generate a random, control-limited LQR as described in the Box-DDP paper.

    Taken from `Control-limited differential dynamic programming`_.

    .. _`Control-limited differential dynamic programming`: https://doi.org/10.1109/ICRA.2014.6907001
    """
    # pylint:enable=line-too-long
    assert 0 < timestep < 1

    state_size = np.random.randint(10, 101)
    ctrl_size = np.random.randint(1, state_size // 2 + 1)

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
