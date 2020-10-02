import numpy as np
import pytest
import torch

from raylab.envs.environments.lqr.generators import box_ddp_random_lqr


@pytest.fixture
def timestep():
    return 0.01


@pytest.fixture
def ctrl_coeff():
    return 0.1


@pytest.fixture
def np_random():
    return np.random.default_rng()


def test_box_ddp_random_lqr(timestep, ctrl_coeff, np_random):
    # pylint:disable=invalid-name
    lqr, _ = box_ddp_random_lqr(timestep, ctrl_coeff, np_random)

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
