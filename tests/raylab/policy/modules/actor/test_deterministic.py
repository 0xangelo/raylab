import warnings
from contextlib import contextmanager, nullcontext

import pytest
import torch
from ray.rllib import SampleBatch


@contextmanager
def suppress_layer_norm_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Separate behavior policy requested and layer normalization deactivated",
            category=UserWarning,
            module="raylab.policy.modules.actor.deterministic",
        )
        yield


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


@pytest.fixture(params=(True, False), ids=lambda x: f"SeparateBehavior({x})")
def separate_behavior(request):
    return request.param


@pytest.fixture
def spec(module_cls, separate_behavior, separate_target_policy):
    return module_cls.spec_cls(
        separate_behavior=separate_behavior,
        separate_target_policy=separate_target_policy,
    )


@pytest.fixture
def module_creator(module_cls, obs_space, action_space):
    return lambda spec: module_cls(obs_space, action_space, spec)


@pytest.fixture
@suppress_layer_norm_warning()
def module(module_creator, spec):
    return module_creator(spec)


def test_module_creation(module_creator, spec):
    if spec.separate_behavior and not spec.network.layer_norm:
        ctx = pytest.warns(UserWarning, match="layer normalization deactivated")
    else:
        ctx = nullcontext()

    with ctx:
        module = module_creator(spec)

    for attr in "policy behavior target_policy".split():
        assert hasattr(module, attr)

    policy, target_policy = module.policy, module.target_policy
    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(policy.parameters(), target_policy.parameters())
    )


@suppress_layer_norm_warning()
def test_separate_behavior(module_cls, obs_space, action_space):
    spec = module_cls.spec_cls(separate_behavior=True)
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
