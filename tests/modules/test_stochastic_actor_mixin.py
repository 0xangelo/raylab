# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.modules.catalog import SACModule, SVGModule


@pytest.fixture(params=(SACModule, SVGModule))
def module_cls(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("InputDepScale", "InputIndepScale"))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("MeanAction", "SampleAction"))
def mean_action_only(request):
    return request.param


@pytest.fixture
def module_batch_fn(module_and_batch_fn, module_cls):
    return partial(module_and_batch_fn, module_cls)


def test_actor_sampler(module_batch_fn, input_dependent_scale, mean_action_only):
    module, batch = module_batch_fn(
        {
            "mean_action_only": mean_action_only,
            "actor": {"input_dependent_scale": input_dependent_scale},
        }
    )
    action = batch[SampleBatch.ACTIONS]

    samples, logp = module.actor.rsample(batch[SampleBatch.CUR_OBS])
    samples_, _ = module.actor.rsample(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == torch.float32
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert samples.dtype == torch.float32
    assert mean_action_only or not torch.allclose(samples, samples_)


def test_actor_params(module_batch_fn, input_dependent_scale):
    module, batch = module_batch_fn(
        {"actor": {"input_dependent_scale": input_dependent_scale}}
    )

    params = module.actor(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale" in params

    loc, scale_diag = params.values()
    action = batch[SampleBatch.ACTIONS]
    assert loc.shape == action.shape
    assert scale_diag.shape == action.shape
    assert loc.dtype == torch.float32
    assert scale_diag.dtype == torch.float32

    pi_params = set(module.actor.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)

    for par in pi_params:
        par.grad = None
    module.actor(batch[SampleBatch.CUR_OBS])["scale"].mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)


def test_actor_reproduce(module_batch_fn, input_dependent_scale):
    module, batch = module_batch_fn(
        {"actor": {"input_dependent_scale": input_dependent_scale}}
    )

    acts = batch[SampleBatch.ACTIONS]
    _acts = module.actor.reproduce(batch[SampleBatch.CUR_OBS], acts)
    assert _acts.shape == acts.shape
    assert _acts.dtype == acts.dtype
    assert torch.allclose(_acts, acts, atol=1e-6)

    _acts.mean().backward()
    pi_params = set(module.actor.parameters())
    assert all(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(module.parameters()) - pi_params)
