# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch

from raylab.utils.replay_buffer import ReplayBuffer


@pytest.fixture(params=[(), ("a",), ("a", "b")])
def extra_keys(request):
    return request.param


@pytest.fixture
def replay_and_keys(extra_keys):
    return ReplayBuffer(size=int(1e4), extra_keys=extra_keys), extra_keys


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
