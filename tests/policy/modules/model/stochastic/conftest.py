# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch


@pytest.fixture
def log_prob_inputs(batch):
    return [
        batch[k]
        for k in (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)
    ]


@pytest.fixture
def sample_inputs(batch):
    return [batch[k] for k in (SampleBatch.CUR_OBS, SampleBatch.ACTIONS)]
