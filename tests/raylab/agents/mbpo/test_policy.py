import pytest


@pytest.fixture(scope="module")
def policy_cls(policy_fn):
    from raylab.agents.mbpo import MBPOTorchPolicy

    def make_policy(config):
        return policy_fn(MBPOTorchPolicy, config)

    return make_policy


@pytest.fixture(scope="module", params=(1, 4), ids=lambda s: f"Ensemble:{s}")
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module", params=(0.0, 0.5, 1.0), ids=lambda x: f"RealData%:{x}")
def real_data_ratio(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size, real_data_ratio):
    return {
        "real_data_ratio": real_data_ratio,
        "model_training": {
            "dataloader": {"batch_size": 32, "replacement": False},
            "max_epochs": 10,
            "max_time": 4,
            "improvement_threshold": 0.01,
            "patience_epochs": 5,
        },
        "model_sampling": {"rollout_schedule": [(0, 10)], "num_elites": 1},
        "module": {"model": {"ensemble_size": ensemble_size}},
    }


@pytest.fixture(scope="module")
def policy(policy_cls, config):
    return policy_cls(config)


def test_policy_creation(policy):
    for attr in "models actor alpha critics".split():
        assert hasattr(policy.module, attr)

    assert "models" in policy.optimizers
    assert "actor" in policy.optimizers
    assert "critics" in policy.optimizers
    assert "alpha" in policy.optimizers

    for attr in "replay virtual_replay".split():
        assert hasattr(policy, attr)
