# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def config():
    return {"module": {"ensemble_size": 1}, "model_rollout_length": 10}


@pytest.fixture(scope="module")
def policy(policy_and_batch_fn, config):
    policy, _ = policy_and_batch_fn(config)
    return policy


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
