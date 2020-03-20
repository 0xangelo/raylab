# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import MAPOModule, SVGModule


@pytest.fixture(params=(MAPOModule, SVGModule))
def module_cls(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("InputDepScale", "InputIndepScale"))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("ResidualModel", "StandardModel"))
def residual(request):
    return request.param


@pytest.fixture
def config(input_dependent_scale, residual):
    return {
        "model": {"input_dependent_scale": input_dependent_scale},
        "residual": residual,
    }


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_model_sampler(module_batch_fn, config):
    module, batch = module_batch_fn(config)

    samples, logp = module.model.sampler(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    samples_, _ = module.model.sampler(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    assert samples.shape == batch[SampleBatch.NEXT_OBS].shape
    assert samples.dtype == torch.float32
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.allclose(samples, samples_)


def test_model_params(module_batch_fn, config):
    module, batch = module_batch_fn(config)

    params = module.model.params(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
    assert "loc" in params
    assert "scale_diag" in params

    loc, scale_diag = params.values()
    assert loc.shape == batch[SampleBatch.NEXT_OBS].shape
    assert scale_diag.shape == batch[SampleBatch.NEXT_OBS].shape
    assert loc.dtype == torch.float32
    assert scale_diag.dtype == torch.float32

    parameters = set(module.model.parameters())
    for par in parameters:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in parameters)
    assert all(p.grad is None for p in set(module.parameters()) - parameters)

    for par in parameters:
        par.grad = None
    module.model.params(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])[
        "scale_diag"
    ].mean().backward()
    assert any(p.grad is not None for p in parameters)
    assert all(p.grad is None for p in set(module.parameters()) - parameters)


def test_model_logp(module_batch_fn, config):
    module, batch = module_batch_fn(config)

    logp = module.model.logp(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.isnan(logp).any()
    assert torch.isfinite(logp).all()


def test_model_reproduce(module_batch_fn, config):
    module, batch = module_batch_fn(config)

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
