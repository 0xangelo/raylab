import pytest
import gym.spaces as spaces

from raylab.algorithms.registry import ALGORITHMS as ALGS
from raylab.utils.debug import fake_batch


@pytest.fixture
def sop_trainer():
    return ALGS["SOP"]()


@pytest.fixture
def sop_policy(sop_trainer):
    return sop_trainer._policy


@pytest.fixture
def cartpole_swingup_env(time_limited_env):
    return lambda _: time_limited_env(
        {"env_id": "CartPoleSwingUp", "time_aware": True, "max_episode_steps": 200}
    )


@pytest.fixture(params=((1,), (2,), (4,)))
def shape(request):
    return request.param


@pytest.fixture
def obs_space(shape):
    return spaces.Box(-10, 10, shape=shape)


@pytest.fixture
def action_space(shape):
    return spaces.Box(-1, 1, shape=shape)


@pytest.fixture
def policy_and_batch_fn(sop_policy, obs_space, action_space):
    def make_policy_and_batch(config):
        policy = sop_policy(obs_space, action_space, config)
        batch = policy._lazy_tensor_dict(
            fake_batch(obs_space, action_space, batch_size=10)
        )
        return policy, batch

    return make_policy_and_batch
