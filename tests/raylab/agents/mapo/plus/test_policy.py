import pytest

from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import MAPO
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import SPAML


@pytest.fixture
def policy_cls():
    from raylab.agents.mapo.plus import MAPOPlusTorchPolicy

    return MAPOPlusTorchPolicy


def test_init(policy_cls, obs_space, action_space):
    policy = policy_cls(obs_space, action_space, {})

    assert isinstance(policy.loss_actor, MAPO)
    assert isinstance(policy.loss_critic, DynaSoftCDQLearning)
    assert isinstance(policy.loss_paml, SPAML)
    assert isinstance(policy.loss_mle, ModelEnsembleMLE)
