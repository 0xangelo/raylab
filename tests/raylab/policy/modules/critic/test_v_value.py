import pytest
import torch

from raylab.policy.modules.actor import Alpha
from raylab.policy.modules.critic import HardValue
from raylab.policy.modules.critic import MLPVValue
from raylab.policy.modules.critic import SoftValue
from raylab.policy.modules.critic import VValue
from raylab.policy.modules.critic import VValueEnsemble


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


@pytest.fixture
def critics(action_critics):
    _, target_critics = action_critics
    target_critics.requires_grad_(True)
    return target_critics


@pytest.fixture
def alpha():
    return Alpha(1.0)


@pytest.fixture
def soft_value(stochastic_policy, critics, alpha):
    return SoftValue(stochastic_policy, critics, alpha)


def test_soft_init(soft_value):
    assert isinstance(soft_value, VValue)


def test_soft_call(soft_value, obs, stochastic_policy, critics, alpha):
    value = soft_value(obs)
    assert value.shape == obs.shape[:-1]
    assert value.grad_fn is not None

    value.sum().backward()
    parameters = set.union(
        set(stochastic_policy.parameters()),
        set(critics.parameters()),
        set(alpha.parameters()),
    )
    assert all([p.grad is not None for p in parameters])


@pytest.fixture
def deterministic_policy(deterministic_policies):
    _, target = deterministic_policies
    return target


@pytest.fixture
def hard_value(deterministic_policy, critics):
    return HardValue(deterministic_policy, critics)


def test_hard_init(hard_value):
    assert isinstance(hard_value, VValue)


def test_hard_call(hard_value, obs, deterministic_policy, critics):
    value = hard_value(obs)
    assert value.shape == obs.shape[:-1]
    assert value.grad_fn is not None

    value.sum().backward()
    parameters = set.union(
        set(deterministic_policy.parameters()), set(critics.parameters())
    )
    assert all([p.grad is not None for p in parameters])
