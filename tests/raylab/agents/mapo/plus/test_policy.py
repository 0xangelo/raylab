import pytest

from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import MAPO
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import SPAML


@pytest.fixture
def policy_cls():
    from raylab.agents.mapo.plus import MAPOPlusTorchPolicy

    return MAPOPlusTorchPolicy


@pytest.fixture
def policy(policy_cls, obs_space, action_space):
    return policy_cls(obs_space, action_space, {})


def test_init(policy_cls, obs_space, action_space):
    policy = policy_cls(obs_space, action_space, {})

    assert isinstance(policy.loss_actor, MAPO)
    assert isinstance(policy.loss_critic, DynaSoftCDQLearning)
    assert isinstance(policy.loss_paml, SPAML)
    assert isinstance(policy.loss_mle, ModelEnsembleMLE)


def test_options(policy):
    options = policy.options
    defaults = options.defaults

    assert "losses" in defaults
    assert "grad_estimator" in defaults["losses"]
    assert "manhattan" in defaults["losses"]
    assert "model_samples" in defaults["losses"]


def test_model_losses(policy):
    assert isinstance(policy.model_training_loss, SPAML)
    assert isinstance(policy.model_warmup_loss, ModelEnsembleMLE)
