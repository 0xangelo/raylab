# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.agents.sop import SOPTorchPolicy


@pytest.fixture(params=(True, False))
def clipped_double_q(request):
    return request.param


@pytest.fixture
def config(clipped_double_q):
    return {"clipped_double_q": clipped_double_q}


@pytest.fixture
def policy(obs_space, action_space, config):
    return SOPTorchPolicy(obs_space, action_space, config)


def test_target_params_update(policy):
    params = list(policy.module.critics.parameters())
    target_params = list(policy.module.target_critics.parameters())
    assert all(torch.allclose(p, q) for p, q in zip(params, target_params))

    old_params = [p.clone() for p in target_params]
    for param in params:
        param.data.add_(torch.ones_like(param))
    policy.update_targets("critics", "target_critics")
    assert all(not torch.allclose(p, q) for p, q in zip(target_params, old_params))
