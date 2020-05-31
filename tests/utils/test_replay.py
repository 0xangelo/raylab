# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import numpy as np
import pytest
from ray.rllib import SampleBatch

from raylab.utils.debug import fake_batch
from raylab.utils.replay_buffer import ReplayBuffer


@pytest.fixture(params=[(), ("a",), ("a", "b")])
def extra_keys(request):
    return request.param


@pytest.fixture
def replay():
    return ReplayBuffer(size=int(1e4))


@pytest.fixture
def sample_batch(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=10)


@pytest.fixture
def replay_and_keys(extra_keys):
    return ReplayBuffer(size=int(1e4), extra_keys=extra_keys), extra_keys


def test_size_zero():
    ReplayBuffer(0)


def test_replay_init(replay_and_keys):
    replay, extra_keys = replay_and_keys

    assert all(
        k in replay._batch_keys
        for k in [
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.NEXT_OBS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
        ]
    )
    assert all(k in replay._batch_keys for k in extra_keys)
    assert replay._maxsize == int(1e4)


def test_all_samples(replay, sample_batch):
    for row in sample_batch.rows():
        replay.add(row)

    buffer = replay.all_samples()
    assert all(np.allclose(sample_batch[k], buffer[k]) for k in sample_batch.keys())
