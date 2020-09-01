import pytest

from raylab.agents.model_based import ModelBasedTrainer


@pytest.fixture
def policy_cls(dummy_policy_cls):
    class DummyPolicy(dummy_policy_cls):
        def set_reward_from_callable(self, _):
            pass

        def set_reward_from_config(self):
            pass

        def set_termination_from_callable(self, _):
            pass

        def set_termination_from_config(self):
            pass

        def set_dynamics_from_callable(self, _):
            pass

        @staticmethod
        def optimize_model(samples, warmup):
            del samples, warmup
            return [], {"model_epochs": 1}

    return DummyPolicy


@pytest.fixture
def trainer_cls(policy_cls):
    class DummyTrainer(ModelBasedTrainer):
        _name = "DummyTrainer"
        _policy = policy_cls

    return DummyTrainer


@pytest.fixture
def config(
    model_update_interval,
    policy_improvement_interval,
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
        model_update_interval=model_update_interval,
        policy_improvement_interval=policy_improvement_interval,
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

    assert trainer.iteration == 0


@pytest.fixture
def trainer(trainer_cls, config):
    return trainer_cls(config=config)


def test_train(mocker, trainer, policy_cls, learning_starts, rollout_fragment_length):
    sample_until_learning_starts = mocker.spy(
        ModelBasedTrainer, "sample_until_learning_starts"
    )
    train_dynamics_model = mocker.spy(ModelBasedTrainer, "train_dynamics_model")
    learn_on_batch = mocker.spy(policy_cls, "learn_on_batch")
    _evaluate = mocker.spy(ModelBasedTrainer, "_evaluate")

    res = trainer.train()
    assert isinstance(res, dict)

    assert "evaluation" in res

    timesteps_this_iter = learning_starts + rollout_fragment_length
    assert res["timesteps_this_iter"] == timesteps_this_iter
    assert len(trainer.replay) == timesteps_this_iter
    assert trainer.metrics.num_steps_sampled == timesteps_this_iter

    assert sample_until_learning_starts.called
    assert train_dynamics_model.called
    assert learn_on_batch.called
    assert _evaluate.called
