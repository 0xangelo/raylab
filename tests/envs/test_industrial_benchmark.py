# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import numpy as np


@pytest.fixture
def classic_reward_ib(envs):
    return envs["IndustrialBenchmark"]({"reward_type": "classic"})


@pytest.fixture
def delta_reward_ib(envs):
    return envs["IndustrialBenchmark"]({"reward_type": "delta"})


def test_reward_type(classic_reward_ib, delta_reward_ib):
    classic_reward_ib.seed(42)
    classic_reward_ib.reset()
    delta_reward_ib.seed(42)
    delta_reward_ib.reset()

    _, rew, _, _ = classic_reward_ib.step([0.5] * 3)
    _, rew2, _, _ = classic_reward_ib.step([0.5] * 3)
    _, _, _, _ = delta_reward_ib.step([0.5] * 3)
    _, rew_, _, _ = delta_reward_ib.step([0.5] * 3)

    assert np.allclose(rew2 - rew, rew_)
