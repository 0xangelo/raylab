import pytest

from raylab.agents.off_policy import OffPolicyMixin
from raylab.agents.trainer import Trainer
from raylab.options import configure


@pytest.fixture
def policy_cls(dummy_policy_cls):
    return dummy_policy_cls


@pytest.fixture
def trainer_cls(policy_cls):
    @configure
    @OffPolicyMixin.add_options
    class Sub(OffPolicyMixin, Trainer):
        _name = "Dummy"
        _policy = policy_cls

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
def config(
    rollout_fragment_length,
    timesteps_per_iteration,
    train_batch_size,
    learning_starts,
):
    # pylint:disable=too-many-arguments
    return {
        "env": "MockEnv",
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": timesteps_per_iteration,
        "train_batch_size": train_batch_size,
        "learning_starts": learning_starts,
    }


@pytest.fixture
def trainer(trainer_cls, config):
    return trainer_cls(config=config)


def test_first_train(
    trainer, timesteps_per_iteration, learning_starts, trainable_info_keys
):
    expected_timesteps = max(timesteps_per_iteration, learning_starts)
    res = trainer.train()

    res_keys = set(res.keys())
    assert all(key in res_keys for key in trainable_info_keys)
    assert res.get("timesteps_total") == expected_timesteps
    assert "timesteps_this_iter" not in res

    policy = trainer.get_policy()
    assert policy.global_timestep == expected_timesteps


def test_second_train(trainer, timesteps_per_iteration, learning_starts):
    expected_timesteps = (
        max(timesteps_per_iteration, learning_starts) + timesteps_per_iteration
    )

    for _ in range(2):
        res = trainer.train()

    assert res["timesteps_total"] == expected_timesteps
    assert "timesteps_this_iter" not in res

    policy = trainer.get_policy()
    assert policy.global_timestep == expected_timesteps
