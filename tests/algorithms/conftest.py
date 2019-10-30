import pytest
import gym.spaces as spaces

from raylab.utils.debug import fake_batch


@pytest.fixture(autouse=True, params=((1,), (2,), (4,)))
def shape(request):
    return request.param


@pytest.fixture
def obs_space(shape):
    return spaces.Box(-10, 10, shape=shape)


@pytest.fixture
def action_space(shape):
    return spaces.Box(-1, 1, shape=shape)


@pytest.fixture
def policy_and_batch_fn(obs_space, action_space):
    def make_policy_and_batch(policy_cls, config):
        policy = policy_cls(obs_space, action_space, config)
        batch = policy._lazy_tensor_dict(
            fake_batch(obs_space, action_space, batch_size=10)
        )
        return policy, batch

    return make_policy_and_batch
