# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import MAPOModule, SVGModule


@pytest.fixture(scope="module", params=(MAPOModule, SVGModule))
def module_cls(request):
    return request.param


@pytest.fixture(
    scope="module", params=(True, False), ids=("InputDepScale", "InputIndepScale")
)
def input_dependent_scale(request):
    return request.param


@pytest.fixture(
    scope="module", params=(True, False), ids=("ResidualModel", "StandardModel")
)
def residual(request):
    return request.param


@pytest.fixture(scope="module")
def config(input_dependent_scale, residual):
    return {
        "model": {"input_dependent_scale": input_dependent_scale},
        "residual": residual,
    }


@pytest.fixture(scope="module")
def module_batch_config(module_and_batch_fn, module_cls, config):
    module, batch = module_and_batch_fn(module_cls, config)
    return module, batch, config


def test_model_rsample(module_batch_config):
    module, batch, _ = module_batch_config

    samples, logp = module.model.rsample(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    samples_, _ = module.model.rsample(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    assert samples.shape == batch[SampleBatch.NEXT_OBS].shape
    assert samples.dtype == torch.float32
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.allclose(samples, samples_)


def test_model_params(module_batch_config):
    module, batch, _ = module_batch_config

    params = module.model(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    assert loc.shape == batch[SampleBatch.NEXT_OBS].shape
    assert scale.shape == batch[SampleBatch.NEXT_OBS].shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    parameters = set(module.model.parameters())
    for par in parameters:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in parameters)
    assert all(p.grad is None for p in set(module.parameters()) - parameters)

    for par in parameters:
        par.grad = None
    module.model(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])[
        "scale"
    ].mean().backward()
    assert any(p.grad is not None for p in parameters)
    assert all(p.grad is None for p in set(module.parameters()) - parameters)


def test_model_logp(module_batch_config):
    module, batch, _ = module_batch_config

    logp = module.model.log_prob(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.isnan(logp).any()
    assert torch.isfinite(logp).all()


def test_model_reproduce(module_batch_config):
    module, batch, _ = module_batch_config

    next_obs = batch[SampleBatch.NEXT_OBS]
    _next_obs = module.model.reproduce(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], next_obs
    )
    assert _next_obs.shape == next_obs.shape
    assert _next_obs.dtype == next_obs.dtype
    assert torch.allclose(_next_obs, next_obs, atol=1e-6)

    _next_obs.mean().backward()
    model_params = set(module.model.parameters())
    assert all(p.grad is not None for p in model_params)
    assert all(p.grad is None for p in set(module.parameters()) - model_params)
