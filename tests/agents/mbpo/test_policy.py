# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

ENSEMBLE_SIZE = (1, 4)


@pytest.fixture(
    scope="module", params=ENSEMBLE_SIZE, ids=(f"Ensemble({s})" for s in ENSEMBLE_SIZE)
)
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size):
    return {
        "model_training": {
            "dataloader": {"batch_size": 32, "replacement": False},
            "max_epochs": 10,
            "max_time": 4,
            "improvement_threshold": 0.01,
            "patience_epochs": 5,
        },
        "model_sampling": {"rollout_length": 10, "num_elites": 1},
        "module": {"ensemble_size": ensemble_size},
    }


@pytest.fixture(scope="module")
def policy(policy_cls, config):
    return policy_cls(config)


def test_policy_creation(policy):
    assert "models" in policy.module
    assert "actor" in policy.module
    assert "critics" in policy.module
    assert "alpha" in policy.module

    assert len(policy.optimizer) == 4
