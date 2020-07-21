import pytest
import torch

from raylab.policy.modules.critic.v_value import MLPVValue
from raylab.policy.modules.critic.v_value import VValueEnsemble


@pytest.fixture(params=(1, 2), ids=lambda x: f"VValues({x})")
def n_critics(request):
    return request.param


@pytest.fixture
def v_value_ensemble(obs_space, n_critics):
    v_values = [MLPVValue(obs_space, MLPVValue.spec_cls()) for _ in range(n_critics)]
    return VValueEnsemble(v_values)


def test_forward(v_value_ensemble, obs, n_critics):
    critics = v_value_ensemble
    values = critics(obs)
    assert torch.is_tensor(values)
    assert values.dtype == torch.float32
    assert values.shape == (len(obs), n_critics)

    clipped = critics(obs).min(dim=-1)[0]
    assert torch.is_tensor(clipped)
    assert clipped.dtype == torch.float32
    assert clipped.shape == (len(obs),)
