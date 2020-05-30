# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch
from ray.rllib.evaluation.metrics import get_learner_stats

from raylab.utils.debug import fake_batch

ENSEMBLE_SIZE = (1, 4)


@pytest.fixture(
    scope="module", params=ENSEMBLE_SIZE, ids=(f"Ensemble({s})" for s in ENSEMBLE_SIZE)
)
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size):
    return {
        "max_model_epochs": 10,
        "model_batch_size": 32,
        "max_model_train_s": 4,
        "improvement_threshold": 0.01,
        "patience_epochs": 5,
        "module": {"ensemble_size": ensemble_size},
        "model_rollout_length": 10,
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


def test_generate_virtual_sample_batch(policy):
    obs_space, action_space = policy.observation_space, policy.action_space
    initial_states = 10
    samples = fake_batch(obs_space, action_space, batch_size=initial_states)
    batch = policy.generate_virtual_sample_batch(samples)

    assert isinstance(batch, SampleBatch)
    assert SampleBatch.CUR_OBS in batch
    assert SampleBatch.ACTIONS in batch
    assert SampleBatch.NEXT_OBS in batch
    assert SampleBatch.REWARDS in batch
    assert SampleBatch.DONES in batch

    total_count = policy.config["model_rollout_length"] * initial_states
    assert batch.count == total_count
    assert batch[SampleBatch.CUR_OBS].shape == (total_count,) + obs_space.shape
    assert batch[SampleBatch.ACTIONS].shape == (total_count,) + action_space.shape
    assert batch[SampleBatch.NEXT_OBS].shape == (total_count,) + obs_space.shape
    assert batch[SampleBatch.REWARDS].shape == (total_count,)
    assert batch[SampleBatch.REWARDS].shape == (total_count,)


def test_optimize_model(policy):
    obs_space, action_space = policy.observation_space, policy.action_space
    train_samples = fake_batch(obs_space, action_space, batch_size=80)
    eval_samples = fake_batch(obs_space, action_space, batch_size=20)

    info = get_learner_stats(policy.optimize_model(train_samples, eval_samples))

    assert "model_epochs" in info
    assert "loss(models)" in info
    assert all(
        f"loss(model[{i}])" in info
        for i in range(policy.config["module"]["model"]["ensemble_size"])
    )
    assert "grad_norm(models)" in info
