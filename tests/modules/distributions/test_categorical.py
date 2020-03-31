# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import Independent, Categorical

from .utils import _test_dist_ops


@pytest.fixture
def categorical(torch_script):
    return torch.jit.script(Categorical()) if torch_script else Categorical()


@pytest.fixture
def independent_categorical(torch_script):
    dist = Independent(Categorical(), reinterpreted_batch_ndims=1)
    return torch.jit.script(dist) if torch_script else dist


def test_categorical(categorical, sample_shape):
    dist = categorical
    flat = torch.randn(4)

    params = dist(flat)
    assert "logits" in params
    batch_shape = params["logits"].shape[:-1]
    event_shape = ()

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)


def test_independent_categorical(independent_categorical, sample_shape):
    dist = independent_categorical
    flat = torch.randn(2, 4)

    params = dist(flat)
    assert "logits" in params
    assert params["logits"].shape == (2, 4)
    batch_shape = ()
    event_shape = (2,)

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)
