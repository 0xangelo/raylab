import pytest
from ray.rllib import Policy
from ray.rllib.agents.trainer import COMMON_CONFIG

from raylab.agents.simple_trainer import SimpleTrainer


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
    class Trainer(SimpleTrainer):
        _name = "Dummy"

        def get_policy_class(self, config):
            return policy_cls

    return Trainer


@pytest.fixture
def timesteps_per_iteration(rollout_fragment_length):
    return 10 * rollout_fragment_length


@pytest.fixture
def train_batch_size():
    return 1


@pytest.fixture
def policy_config():
    return {}


@pytest.fixture
def wandb_config():
    return {}


@pytest.fixture
def config(
    rollout_fragment_length,
    timesteps_per_iteration,
    train_batch_size,
    policy_config,
    wandb_config,
):
    return {
        "env": "MockEnv",
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": timesteps_per_iteration,
        "train_batch_size": train_batch_size,
        "policy": policy_config,
        "wandb": wandb_config,
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
    assert "wandb" in trainer.config


def test_train(trainer, timesteps_per_iteration, trainable_info_keys):
    res = trainer.train()

    res_keys = set(res.keys())
    assert all(key in res_keys for key in trainable_info_keys)
    assert res.get("timesteps_total") == timesteps_per_iteration
    assert "timesteps_this_iter" not in res
