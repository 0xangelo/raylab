# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import Independent, Normal

from .utils import _test_dist_ops


@pytest.fixture
def normal(torch_script):
    return torch.jit.script(Normal()) if torch_script else Normal()


@pytest.fixture
def independent_normal(torch_script):
    dist = Independent(Normal(), reinterpreted_batch_ndims=1)
    return torch.jit.script(dist) if torch_script else dist


def test_normal(normal, sample_shape):
    dist = normal
    flat = torch.stack([torch.zeros([]), torch.ones([])], dim=0)

    params = dist(flat)
    assert "loc" in params
    assert "scale" in params
    batch_shape = params["loc"].shape
    event_shape = ()

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)


def test_independent_normal(independent_normal, sample_shape):
    dist = independent_normal
    flat = torch.cat([torch.zeros(2), torch.ones(2)], dim=0)

    params = dist(flat)
    assert "loc" in params
    assert "scale" in params
    assert params["loc"].shape == (2,)
    assert params["scale"].shape == (2,)
    batch_shape = params["loc"].shape[:-1]
    event_shape = params["loc"].shape[-1:]

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)
