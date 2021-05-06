import pytest
from ray.rllib import Policy
from ray.rllib.agents.trainer import COMMON_CONFIG

from raylab.agents.trainer import Trainer


@pytest.fixture
def policy_cls(dummy_policy_cls):
    class CLS(dummy_policy_cls):
        # pylint:disable=all
        compiled: bool = False

        def compile(self):
            self.compiled = True

    return CLS


@pytest.fixture
def trainer_cls(policy_cls):
    class Sub(Trainer):
        _name = "Dummy"
        _policy_class = policy_cls

    return Sub


@pytest.fixture
def timesteps_per_iteration(rollout_fragment_length):
    return 10 * rollout_fragment_length


@pytest.fixture
def train_batch_size():
    return 1


@pytest.fixture(params=(0, 1000), ids=lambda x: f"LearningStarts:{x}")
def learning_starts(request):
    return request.param


@pytest.fixture
def policy_config():
    return {}


@pytest.fixture
def config(
    rollout_fragment_length,
    timesteps_per_iteration,
    train_batch_size,
    policy_config,
):
    # pylint:disable=too-many-arguments
    return {
        "env": "MockEnv",
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": timesteps_per_iteration,
        "train_batch_size": train_batch_size,
        "policy": policy_config,
    }


def test_optimize_policy_backend(trainer_cls, config):
    config = {**config, "policy": {"compile": True}}
    trainer = trainer_cls(config=config)
    assert trainer.get_policy().compiled


@pytest.fixture
def trainer(trainer_cls, config):
    return trainer_cls(config=config)


def test_policy(trainer):
    policy = trainer.get_policy()
    assert isinstance(policy, Policy)


def test_config(trainer):
    assert set(COMMON_CONFIG.keys()).issubset(set(trainer.config.keys()))
    assert "policy" in trainer.config


def test_first_train(trainer, timesteps_per_iteration, trainable_info_keys):
    expected_timesteps = timesteps_per_iteration
    res = trainer.train()

    res_keys = set(res.keys())
    assert all(key in res_keys for key in trainable_info_keys)
    assert res.get("timesteps_total") == expected_timesteps
    assert "timesteps_this_iter" not in res

    policy = trainer.get_policy()
    assert policy.global_timestep == expected_timesteps


def test_second_train(trainer, timesteps_per_iteration):
    expected_timesteps = timesteps_per_iteration * 2

    for _ in range(2):
        res = trainer.train()

    assert res["timesteps_total"] == expected_timesteps
    assert "timesteps_this_iter" not in res

    policy = trainer.get_policy()
    assert policy.global_timestep == expected_timesteps
