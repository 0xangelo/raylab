import numpy as np
import pytest
import torch

from raylab.envs.environments.lqr.generators import box_ddp_random_lqr
from raylab.envs.environments.lqr.generators import make_lqr
from raylab.envs.environments.lqr.generators import make_lqr_linear_navigation
from raylab.envs.environments.lqr.types import LQR


@pytest.fixture
def timestep():
    return 0.01


@pytest.fixture
def ctrl_coeff():
    return 0.1


@pytest.fixture
def np_random():
    return np.random.default_rng()


def check_lqr_mats(lqr: LQR):
    # pylint:disable=invalid-name
    assert all([torch.is_tensor(t) for t in lqr])

    F, f, C, c = lqr
    dim = F.shape[1]
    x_size = F.shape[0]

    assert F.shape == (x_size, dim)
    assert f.shape == (x_size,)
    assert C.shape == (dim, dim)
    assert c.shape == (dim,)

    eigv, _ = torch.eig(C)
    assert eigv.ge(0).all()


def test_box_ddp_random_lqr(timestep, ctrl_coeff, np_random):
    lqr, _ = box_ddp_random_lqr(timestep, ctrl_coeff, np_random)
    check_lqr_mats(lqr)


@pytest.fixture
def state_size():
    return 10


@pytest.fixture
def ctrl_size():
    return 3


def test_make_lqr(state_size, ctrl_size, np_random):
    lqr = make_lqr(state_size, ctrl_size, np_random)
    check_lqr_mats(lqr)


@pytest.fixture
def goal():
    return (0.5, 1.0)


@pytest.fixture
def beta(ctrl_coeff):
    return ctrl_coeff


def test_make_lqr_linear_navigation(goal, beta):
    lqr, _ = make_lqr_linear_navigation(goal, beta)
    check_lqr_mats(lqr)
