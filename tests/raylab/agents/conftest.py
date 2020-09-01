import itertools

import pytest
from ray.rllib import Policy

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def dummy_policy_cls():
    class DummyPolicy(Policy):
        # pylint:disable=abstract-method,too-many-arguments
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.param = 0
            self.param_seq = itertools.count()
            next(self.param_seq)

            self.exploration = self._create_exploration()

        def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            explore=None,
            timestep=None,
            **kwargs,
        ):
            return [self.action_space.sample() for _ in obs_batch], [], {}

        def learn_on_batch(self, _):
            self.param = next(self.param_seq)
            return {"improved": True}

        def get_weights(self):
            return {"param": self.param}

        def set_weights(self, weights):
            self.param = weights["param"]

    return DummyPolicy


@pytest.fixture(scope="module")
def policy_and_batch_(obs_space, action_space):
    def make_policy_and_batch(policy_cls, config):
        policy = policy_cls(obs_space, action_space, config)
        batch = policy.lazy_tensor_dict(
            fake_batch(obs_space, action_space, batch_size=10)
        )
        return policy, batch

    return make_policy_and_batch


@pytest.fixture(scope="module", params="MockEnv Navigation".split())
def env_name(request):
    return request.param


@pytest.fixture(scope="module")
def policy_fn(env_name):
    from raylab.envs import get_env_creator

    env = get_env_creator(env_name)({})

    def make_policy(policy_cls, config):
        config["env"] = env_name
        return policy_cls(env.observation_space, env.action_space, config)

    return make_policy
