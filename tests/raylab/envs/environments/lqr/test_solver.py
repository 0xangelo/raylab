import numpy as np
import pytest
import torch

from raylab.envs.environments.lqr.generators import box_ddp_random_lqr
from raylab.envs.environments.lqr.solver import LQRSolver


@pytest.fixture(params=(True, False), ids=lambda x: f"TorchScript:{x}")
def solver(request):
    script = request.param
    solvr = LQRSolver()
    return torch.jit.script(solvr) if script else solvr


@pytest.fixture
def lqr():
    gen = np.random.default_rng()
    system, _ = box_ddp_random_lqr(timestep=0.01, ctrl_coeff=0.1, np_random=gen)
    return system


@pytest.fixture
def horizon():
    return 20


def test_forward(solver, lqr, horizon):
    policy, value = solver(lqr, horizon)

    def is_tensor(val):
        return torch.is_tensor(val)

    assert isinstance(policy, list)
    assert all([is_tensor(K) and is_tensor(k) for K, k in policy])

    assert isinstance(value, list)
    assert all([is_tensor(V) and is_tensor(v) and is_tensor(c) for V, v, c in value])
