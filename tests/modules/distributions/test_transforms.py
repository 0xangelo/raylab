# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import (
    AffineTransform,
    ComposeTransform,
    Independent,
    InvTransform,
    Normal,
    SigmoidTransform,
    TanhTransform,
    TransformedDistribution,
    Uniform,
)


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
    )
)
def transform(request, torch_script):
    trans = request.param()
    return torch.jit.script(trans) if torch_script else trans


@pytest.fixture(
    params=(
        lambda: InvTransform(
            AffineTransform(torch.ones(2), torch.ones(2) * 2, event_dim=1)
        ),
        lambda: InvTransform(SigmoidTransform(event_dim=1)),
        lambda: InvTransform(TanhTransform(event_dim=1)),
    )
)
def inv_transform(request):
    return request.param()


def test_transformed_distribution(dist_params, sample_shape, torch_script):
    base_dist, params = dist_params
    dist = TransformedDistribution(
        base_dist,
        ComposeTransform(
            [
                TanhTransform(event_dim=1),
                AffineTransform(-torch.ones(2), torch.ones(2), event_dim=1),
            ]
        ),
    )
    dist = torch.jit.script(dist) if torch_script else dist

    event_shape = (2,)
    batch_shape = ()
    rsample, log_prob = dist.rsample(params, sample_shape)
    assert rsample.shape == sample_shape + batch_shape + event_shape
    assert log_prob.shape == sample_shape + batch_shape

    log_prob = dist.log_prob(params, rsample)
    assert log_prob.shape == sample_shape + batch_shape


def test_transforms(dist_params, transform, sample_shape):
    dist, params = dist_params

    rsample, _ = dist.rsample(params, sample_shape)
    encoded, log_det = transform(rsample)
    assert log_det.shape == sample_shape
    decoded, log_det = transform(encoded, reverse=True)
    assert log_det.shape == sample_shape
    assert torch.allclose(decoded, rsample, atol=1e-7)


def test_inv_transforms(inv_transform):
    torch.jit.script(inv_transform)
