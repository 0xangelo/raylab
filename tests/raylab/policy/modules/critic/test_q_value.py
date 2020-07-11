import pytest
import torch
from ray.rllib import SampleBatch

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


def test_forward(q_value_ensemble, batch, n_critics):
    critics = q_value_ensemble
    obs, act = batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    values = critics(obs, act)
    assert torch.is_tensor(values)
    assert values.dtype == torch.float32
    assert values.shape == (len(obs), n_critics)

    clipped = critics(obs, act, clip=True)
    assert torch.is_tensor(clipped)
    assert clipped.dtype == torch.float32
    assert clipped.shape == (len(obs), n_critics)
