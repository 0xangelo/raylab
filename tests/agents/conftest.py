# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def policy_and_batch_(obs_space, action_space):
    def make_policy_and_batch(policy_cls, config):
        policy = policy_cls(obs_space, action_space, config)
        batch = policy._lazy_tensor_dict(
            fake_batch(obs_space, action_space, batch_size=10)
        )
        return policy, batch

    return make_policy_and_batch


@pytest.fixture(scope="module", params="MockEnv Navigation".split())
def env_name(request):
    return request.param


@pytest.fixture(scope="module")
def policy_fn(envs, env_name):
    env = envs[env_name]({})

    def make_policy(policy_cls, config):
        config["env"] = env_name
        return policy_cls(env.observation_space, env.action_space, config)

    return make_policy
