import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    from raylab.utils.debug import fake_batch

    samples = fake_batch(obs_space, action_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def obs(batch):
    return batch[SampleBatch.CUR_OBS]


@pytest.fixture
def action(batch):
    return batch[SampleBatch.ACTIONS]


@pytest.fixture
def next_obs(batch):
    return batch[SampleBatch.NEXT_OBS]
