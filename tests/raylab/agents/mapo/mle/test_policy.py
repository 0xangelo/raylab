import pytest

from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import MAPO
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import SoftCDQLearning


@pytest.fixture
def policy_cls():
    from raylab.agents.mapo.mle import MlMAPOTorchPolicy

    return MlMAPOTorchPolicy


@pytest.fixture(params=(True, False), ids="DynaQ NormalQ".split())
def dyna_q(request):
    return request.param


def test_init(policy_cls, obs_space, action_space, dyna_q):
    policy = policy_cls(obs_space, action_space, {"losses": {"dyna_q": dyna_q}})

    assert isinstance(policy.loss_mle, ModelEnsembleMLE)
    assert isinstance(policy.loss_actor, MAPO)
    if dyna_q:
        assert isinstance(policy.loss_critic, DynaSoftCDQLearning)
    else:
        assert isinstance(policy.loss_critic, SoftCDQLearning)

    assert policy.model_warmup_loss is policy.loss_mle
    assert policy.model_training_loss is policy.loss_mle
