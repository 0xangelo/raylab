import pytest

from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import ReparameterizedSoftPG


@pytest.fixture(scope="module")
def policy_cls():
    from raylab.agents.sac.dyna import DynaSACTorchPolicy

    return DynaSACTorchPolicy


@pytest.fixture(scope="module")
def module_type():
    return "ModelBasedSAC"


@pytest.fixture(scope="module", params=(1, 4), ids=lambda x: f"Models({x})")
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module")
def config(module_type, ensemble_size):
    return {"module": {"type": module_type, "model": {"ensemble_size": ensemble_size}}}


@pytest.fixture(scope="module")
def policy(policy_cls, obs_space, action_space, config):
    return policy_cls(obs_space, action_space, config)


def test_init(policy, ensemble_size):
    for attr in "models actor critics".split():
        assert hasattr(policy.module, attr)
        assert attr in policy.optimizers
    assert len(policy.module.models) == ensemble_size

    assert isinstance(policy.loss_model, ModelEnsembleMLE)
    assert isinstance(policy.loss_actor, ReparameterizedSoftPG)
    assert isinstance(policy.loss_critic, DynaSoftCDQLearning)
