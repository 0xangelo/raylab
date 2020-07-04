# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def action_space(cont_space):
    return cont_space


@pytest.fixture
def batch(cont_batch):
    return cont_batch


@pytest.fixture(scope="module")
def module_cls():
    from raylab.policy.modules.actor.deterministic import DeterministicActor

    return DeterministicActor


@pytest.fixture(params=(True, False), ids=lambda x: f"SeparateTargetPolicy({x})")
def separate_target_policy(request):
    return request.param


@pytest.fixture(params=(True, False), ids=lambda x: f"ParameterNoise({x})")
def parameter_noise(request):
    return request.param


@pytest.fixture
def spec(module_cls, parameter_noise, separate_target_policy):
    return module_cls.spec_cls(
        parameter_noise=parameter_noise, separate_target_policy=separate_target_policy
    )


@pytest.fixture
def module(module_cls, obs_space, action_space, spec):
    return module_cls(obs_space, action_space, spec)


def test_module_creation(module):
    for attr in "policy behavior target_policy".split():
        assert hasattr(module, attr)

    policy, target_policy = module.policy, module.target_policy
    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(policy.parameters(), target_policy.parameters())
    )


def test_parameter_noise(module_cls, obs_space, action_space):
    spec = module_cls.spec_cls(parameter_noise=True)
    module = module_cls(obs_space, action_space, spec)

    assert all(
        torch.allclose(p, n)
        for p, n in zip(module.policy.parameters(), module.behavior.parameters())
    )


def test_separate_target_policy(module, spec):
    policy, target = module.policy, module.target_policy

    if spec.separate_target_policy:
        assert all(p is not t for p, t in zip(policy.parameters(), target.parameters()))
    else:
        assert all(p is t for p, t in zip(policy.parameters(), target.parameters()))


def test_behavior(module, batch):
    action = batch[SampleBatch.ACTIONS]

    samples = module.behavior(batch[SampleBatch.CUR_OBS])
    samples_ = module.behavior(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == torch.float32
    assert torch.allclose(samples, samples_)
