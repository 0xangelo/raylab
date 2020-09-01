import pytest

from raylab.agents.off_policy import OffPolicyTrainer


@pytest.fixture
def trainer_cls(dummy_policy_cls):
    class DummyTrainer(OffPolicyTrainer):
        _name = "DummyTrainer"
        _policy = dummy_policy_cls

    return DummyTrainer


@pytest.fixture
def config(
    policy_improvements,
    learning_starts,
    train_batch_size,
    rollout_fragment_length,
    num_workers,
    buffer_size,
    timesteps_per_iteration,
    evaluation_interval,
):
    # pylint:disable=too-many-arguments
    return dict(
        env="MockEnv",
        policy_improvements=policy_improvements,
        learning_starts=learning_starts,
        train_batch_size=train_batch_size,
        rollout_fragment_length=rollout_fragment_length,
        num_workers=num_workers,
        buffer_size=buffer_size,
        timesteps_per_iteration=timesteps_per_iteration,
        evaluation_interval=evaluation_interval,
    )


def test_init(trainer_cls, config):
    trainer = trainer_cls(config=config)

    metrics = trainer.metrics
    global_vars = trainer.global_vars
    policy = trainer.get_policy()
    assert (
        metrics.num_steps_sampled
        == global_vars["timestep"]
        == policy.global_timestep
        == 0
    )

    assert hasattr(trainer, "evaluation_workers")
    assert trainer.iteration == 0
    assert trainer._iteration == 0


@pytest.fixture
def trainer(trainer_cls, config):
    return trainer_cls(config=config)


def test_update_steps_sampled(trainer):
    steps = 10
    trainer.update_steps_sampled(steps)

    assert trainer.metrics.num_steps_sampled == steps
    assert trainer.global_vars["timestep"] == steps
    assert trainer.get_policy().global_timestep == steps


def test_train(mocker, trainer, learning_starts, rollout_fragment_length):
    sample_until_learning_starts = mocker.spy(
        OffPolicyTrainer, "sample_until_learning_starts"
    )
    _evaluate = mocker.spy(OffPolicyTrainer, "_evaluate")

    res = trainer.train()
    assert trainer.iteration == 1
    assert trainer._iteration == 1
    assert isinstance(res, dict)
    assert res["timesteps_this_iter"] == learning_starts + rollout_fragment_length
    assert "evaluation" in res

    assert sample_until_learning_starts.called
    assert _evaluate.called
