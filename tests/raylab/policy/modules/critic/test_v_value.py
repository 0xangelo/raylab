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


def _test_value(value, obs):
    assert torch.is_tensor(value)
    assert value.dtype == torch.float32
    assert value.shape == (len(obs),)


def test_forward(v_value_ensemble, obs, n_critics):
    critics = v_value_ensemble
    values = critics(obs)

    assert isinstance(values, list)
    assert len(values) == n_critics
    for value in values:
        _test_value(value, obs)

    clipped = VValueEnsemble.clipped(values)
    _test_value(clipped, obs)
