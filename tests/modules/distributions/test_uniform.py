# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import Independent, Uniform

from .utils import _test_dist_ops


@pytest.fixture
def uniform(torch_script):
    return torch.jit.script(Uniform()) if torch_script else Uniform()


@pytest.fixture
def independent_uniform(torch_script):
    dist = Independent(Uniform(), reinterpreted_batch_ndims=1)
    return torch.jit.script(dist) if torch_script else dist


def test_uniform(uniform, sample_shape):
    dist = uniform
    flat = torch.stack([torch.zeros([]), torch.ones([])], dim=0)

    params = dist(flat)
    assert "low" in params
    assert "high" in params
    batch_shape = params["low"].shape
    event_shape = ()

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)


def test_independent_uniform(independent_uniform, sample_shape):
    dist = independent_uniform
    flat = torch.cat([torch.zeros(2), torch.ones(2)], dim=0)

    params = dist(flat)
    assert "low" in params
    assert "high" in params
    assert params["low"].shape == (2,)
    assert params["high"].shape == (2,)
    batch_shape = params["low"].shape[:-1]
    event_shape = params["low"].shape[-1:]

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)
