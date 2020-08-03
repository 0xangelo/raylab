import pytest
import torch

from raylab.policy.modules.critic.q_value import MLPQValue
from raylab.policy.modules.critic.q_value import QValueEnsemble


@pytest.fixture(params=(1, 2), ids=lambda x: f"QValues({x})")
def n_critics(request):
    return request.param


@pytest.fixture
def q_value_ensemble(obs_space, action_space, n_critics):
    q_values = [
        MLPQValue(obs_space, action_space, MLPQValue.spec_cls())
        for _ in range(n_critics)
    ]
    return QValueEnsemble(q_values)


def test_forward(q_value_ensemble, obs, action, n_critics):
    critics = q_value_ensemble
    values = critics(obs, action)
    assert torch.is_tensor(values)
    assert values.dtype == torch.float32
    assert values.shape == (len(obs), n_critics)

    clipped = critics(obs, action).min(dim=-1)[0]
    assert torch.is_tensor(clipped)
    assert clipped.dtype == torch.float32
    assert clipped.shape == (len(obs),)


def test_script_backprop(q_value_ensemble, obs, action):
    critics = torch.jit.script(q_value_ensemble)
    values, _ = critics(obs, action).min(dim=-1)
    values.mean().backward()
