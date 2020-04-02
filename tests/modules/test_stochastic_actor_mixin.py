# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch
from gym.spaces import Box, Discrete
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tracking_dict import UsageTrackingDict

from raylab.utils.debug import fake_batch
from raylab.utils.pytorch import convert_to_tensor

from raylab.modules.catalog import SACModule, SVGModule, TRPOModule
from raylab.modules.action_value_mixin import ActionValueMixin
from raylab.modules.stochastic_model_mixin import StochasticModelMixin


@pytest.fixture(
    params=(Box(-1, 1, shape=(1,)), Box(-1, 1, shape=(3,)), Discrete(2), Discrete(8),),
    ids=tuple("Box((1,)) Box((3,)) Discrete(2) Discrete(8)".split()),
)
def action_space(request):
    return request.param


@pytest.fixture(params=(SACModule, SVGModule, TRPOModule))
def module_cls(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("InputDepScale", "InputIndepScale"))
def input_dependent_scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("MeanAction", "SampleAction"))
def mean_action_only(request):
    return request.param


@pytest.fixture
def module_batch_spaces_fn(module_cls, obs_space, action_space, torch_script):
    is_discrete = isinstance(action_space, Discrete)
    if issubclass(module_cls, (ActionValueMixin, StochasticModelMixin)) and is_discrete:
        pytest.skip(
            "ActionValueMixin and StochasticModelMixin are currently incompatible with "
            "discrete action spaces"
        )

    def make_module_batch_spaces(config):
        config["torch_script"] = torch_script
        module = module_cls(obs_space, action_space, config)
        module = torch.jit.script(module) if torch_script else module

        batch = UsageTrackingDict(fake_batch(obs_space, action_space, batch_size=10))
        batch.set_get_interceptor(partial(convert_to_tensor, device="cpu"))

        return module, batch, obs_space, action_space

    return make_module_batch_spaces


def test_actor_sampler(module_batch_spaces_fn, input_dependent_scale, mean_action_only):
    module, batch, _, action_space = module_batch_spaces_fn(
        {
            "mean_action_only": mean_action_only,
            "actor": {"input_dependent_scale": input_dependent_scale},
        }
    )
    action = batch[SampleBatch.ACTIONS]

    sampler = (
        module.actor.rsample if isinstance(action_space, Box) else module.actor.sample
    )
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert mean_action_only or not torch.allclose(samples, samples_)


def test_actor_params(module_batch_spaces_fn, input_dependent_scale):
    module, batch, _, action_space = module_batch_spaces_fn(
        {"actor": {"input_dependent_scale": input_dependent_scale}}
    )

    params = module.actor(batch[SampleBatch.CUR_OBS])
    if isinstance(action_space, Box):
        assert "loc" in params
        assert "scale" in params

        loc, scale = params["loc"], params["scale"]
        action = batch[SampleBatch.ACTIONS]
        assert loc.shape == action.shape
        assert scale.shape == action.shape
        assert loc.dtype == torch.float32
        assert scale.dtype == torch.float32

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
    else:
        assert "logits" in params
        logits = params["logits"]
        assert logits.shape[-1] == action_space.n

        pi_params = set(module.actor.parameters())
        for par in pi_params:
            par.grad = None
        logits.mean().backward()
        assert any(p.grad is not None for p in pi_params)
        assert all(p.grad is None for p in set(module.parameters()) - pi_params)


def test_actor_reproduce(module_batch_spaces_fn, input_dependent_scale):
    module, batch, _, action_space = module_batch_spaces_fn(
        {"actor": {"input_dependent_scale": input_dependent_scale}}
    )

    acts = batch[SampleBatch.ACTIONS]
    _acts = module.actor.reproduce(batch[SampleBatch.CUR_OBS], acts)
    if isinstance(action_space, Discrete):
        assert torch.isnan(_acts).all()
    else:
        assert _acts.shape == acts.shape
        assert _acts.dtype == acts.dtype
        assert torch.allclose(_acts, acts, atol=1e-6)

        _acts.mean().backward()
        pi_params = set(module.actor.parameters())
        assert all(p.grad is not None for p in pi_params)
        assert all(p.grad is None for p in set(module.parameters()) - pi_params)
