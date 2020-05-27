# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.modules.mixins import StochasticModelMixin


class DummyModule(StochasticModelMixin, nn.ModuleDict):
    # pylint:disable=abstract-method
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_model(obs_space, action_space, config))


@pytest.fixture(scope="module", params=(DummyModule,))
def module_cls(request):
    return request.param


ENSEMBLE_SIZE = (0, 1, 2)
RESIDUAL = (True, False)
INPUT_DEPENDENT_SCALE = (True, False)


@pytest.fixture(
    scope="module",
    params=ENSEMBLE_SIZE,
    ids=(f"EnsembleSize({s})" for s in ENSEMBLE_SIZE),
)
def ensemble_size(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=INPUT_DEPENDENT_SCALE,
    ids=(f"InputDepScale({i})" for i in INPUT_DEPENDENT_SCALE),
)
def input_dependent_scale(request):
    return request.param


@pytest.fixture(
    scope="module", params=(True, False), ids=(f"Residual({r})" for r in RESIDUAL)
)
def residual(request):
    return request.param


@pytest.fixture(scope="module")
def config(input_dependent_scale, residual):
    return {
        "model": {
            "residual": residual,
            "input_dependent_scale": input_dependent_scale,
            "ensemble_size": 0,
        },
    }


@pytest.fixture(scope="module")
def module_batch_config(module_and_batch_fn, module_cls, config):
    module, batch = module_and_batch_fn(module_cls, config)
    return module, batch, config


def test_ensemble(module_and_batch_fn, module_cls, config, ensemble_size):
    config["model"]["ensemble_size"] = ensemble_size
    module, _ = module_and_batch_fn(module_cls, config)

    if ensemble_size:
        assert "models" in module
        assert len(module.models) == ensemble_size
    else:
        assert "model" in module


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
    next_obs_, logp_ = module.model.reproduce(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], next_obs
    )
    assert next_obs_.shape == next_obs.shape
    assert next_obs_.dtype == next_obs.dtype
    assert torch.allclose(next_obs_, next_obs, atol=1e-5)
    assert logp_.shape == batch[SampleBatch.REWARDS].shape

    next_obs_.mean().backward()
    model_params = set(module.model.parameters())
    assert all(p.grad is not None for p in model_params)
    assert all(p.grad is None for p in set(module.parameters()) - model_params)
