import pytest
from ray.rllib import SampleBatch


@pytest.fixture
def obs(batch):
    return batch[SampleBatch.CUR_OBS]


@pytest.fixture
def act(batch):
    return batch[SampleBatch.ACTIONS]


@pytest.fixture
def next_obs(batch):
    return batch[SampleBatch.NEXT_OBS]


@pytest.fixture
def rew(batch):
    return batch[SampleBatch.REWARDS]
