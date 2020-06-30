# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture(scope="module")
def module_cls():
    from raylab.modules.networks.actor.policy.deterministic import (
        MLPDeterministicPolicy,
    )

    return MLPDeterministicPolicy


@pytest.fixture(params=(0.1, 1.2), ids=lambda x: f"NormBeta({x})")
def norm_beta(request):
    return request.param


@pytest.fixture
def spec(module_cls):
    return module_cls.spec_cls()


@pytest.fixture
def action_space(cont_space):
    return cont_space


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, norm_beta):
    return module_cls(obs_space, action_space, spec, norm_beta)


def test_unconstrained_action(module, batch, action_space, norm_beta):
    action_dim = action_space.shape[0]

    policy_out = module.unconstrained_action(batch[SampleBatch.CUR_OBS])
    norms = policy_out.norm(p=1, dim=-1, keepdim=True) / action_dim
    assert policy_out.shape[-1] == action_dim
    assert policy_out.dtype == torch.float32
    assert (norms <= (norm_beta + torch.finfo(torch.float32).eps)).all()
