# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from gym.spaces import Box
from gym.spaces import Discrete

from raylab.utils.debug import fake_batch


DISC_SPACES = (Discrete(2), Discrete(8))
CONT_SPACES = (Box(-1, 1, shape=(1,)), Box(-1, 1, shape=(3,)))
ACTION_SPACES = CONT_SPACES + DISC_SPACES


@pytest.fixture(params=DISC_SPACES, ids=(repr(a) for a in DISC_SPACES))
def disc_space(request):
    return request.param


@pytest.fixture(params=CONT_SPACES, ids=(repr(a) for a in CONT_SPACES))
def cont_space(request):
    return request.param


@pytest.fixture(params=ACTION_SPACES, ids=(repr(a) for a in ACTION_SPACES))
def action_space(request):
    return request.param


@pytest.fixture
def disc_batch(obs_space, disc_space):
    samples = fake_batch(obs_space, disc_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def cont_batch(obs_space, cont_space):
    samples = fake_batch(obs_space, cont_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}
