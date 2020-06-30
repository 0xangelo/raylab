# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import numpy as np
import pytest
from ray.rllib import SampleBatch

from raylab.utils.debug import fake_batch
from raylab.utils.replay_buffer import ListReplayBuffer
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.replay_buffer import ReplayField


@pytest.fixture(params=(ListReplayBuffer, NumpyReplayBuffer))
def replay_cls(request, obs_space, action_space):
    cls = request.param
    if issubclass(cls, NumpyReplayBuffer):
        return partial(cls, obs_space=obs_space, action_space=action_space)
    return cls


@pytest.fixture(params=[(), ("a",), ("a", "b")])
def extra_fields(request):
    return (ReplayField(n) for n in request.param)


@pytest.fixture
def replay(replay_cls):
    return replay_cls(size=int(1e4))


@pytest.fixture
def sample_batch(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=10)


@pytest.fixture
def replay_and_fields(replay, extra_fields):
    replay.add_fields(*extra_fields)
    return replay, extra_fields


def test_size_zero(replay_cls):
    replay_cls(size=0)


def test_replay_init(replay_and_fields):
    replay, extra_fields = replay_and_fields

    assert all(
        k in {f.name for f in replay.fields}
        for k in [
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.NEXT_OBS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
        ]
    )
    assert all(f in replay.fields for f in extra_fields)
    assert replay._maxsize == int(1e4)


def test_all_samples(replay, sample_batch):
    for row in sample_batch.rows():
        replay.add(row)

    buffer = replay.all_samples()
    assert all(np.allclose(sample_batch[k], buffer[k]) for k in sample_batch.keys())
