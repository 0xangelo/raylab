# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import (
    AffineTransform,
    CompositeTransform,
    Independent,
    InverseTransform,
    Normal,
    SigmoidTransform,
    TanhTransform,
    TanhSquashTransform,
    TransformedDistribution,
    Uniform,
)

from .utils import _test_dist_ops


@pytest.fixture(params=(Uniform, Normal))
def dist_params(request):
    dist = Independent(request.param(), reinterpreted_batch_ndims=1)
    tensor = torch.cat([torch.zeros(2), torch.ones(2)], dim=0)
    return dist, dist(tensor)


@pytest.fixture(
    params=(
        lambda: AffineTransform(torch.ones(2), torch.ones(2) * 2, event_dim=1),
        lambda: SigmoidTransform(event_dim=1),
        lambda: TanhTransform(event_dim=1),
        lambda: TanhSquashTransform(-2 * torch.ones(2), 2 * torch.ones(2), event_dim=1),
    )
)
def transform(request):
    return request.param()


@pytest.fixture
def inv_transform(transform):
    return InverseTransform(transform)


def test_transformed_distribution(dist_params, sample_shape, torch_script):
    base_dist, params = dist_params
    dist = TransformedDistribution(
        base_dist,
        CompositeTransform(
            [TanhTransform(), AffineTransform(-torch.ones(2), torch.ones(2))],
            event_dim=1,
        ),
    )
    dist = torch.jit.script(dist) if torch_script else dist

    event_shape = (2,)
    batch_shape = ()

    _test_dist_ops(dist, params, batch_shape, event_shape, sample_shape)


def test_transforms(dist_params, transform, sample_shape, torch_script):
    dist, params = dist_params
    transform = torch.jit.script(transform) if torch_script else transform

    rsample, _ = dist.rsample(params, sample_shape)
    encoded, log_det = transform(rsample)
    assert log_det.shape == sample_shape
    decoded, log_det = transform(encoded, reverse=True)
    assert log_det.shape == sample_shape
    assert torch.allclose(decoded, rsample, atol=1e-6)


def test_inv_transforms(inv_transform, torch_script):
    transform = inv_transform
    transform = torch.jit.script(transform) if torch_script else transform

    inputs = torch.rand(10, 2)
    encoded, log_det = transform(inputs, {})
    assert log_det.shape == inputs.shape[: -transform.event_dim]
    decoded, log_det = transform(encoded, {}, reverse=True)
    assert log_det.shape == inputs.shape[: -transform.event_dim]
    assert torch.allclose(decoded, inputs, atol=1e-6)
