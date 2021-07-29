from __future__ import annotations

import itertools
from typing import Callable

import pytest
from ray.rllib import Policy
from ray.tune.logger import Logger, UnifiedLogger

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def logger_creator(tmpdir_factory) -> Callable[[dict], Logger]:
    def creator(config: dict) -> UnifiedLogger:
        logdir = str(tmpdir_factory.mktemp("results"))
        return UnifiedLogger(config, logdir, loggers=None)

    return creator


@pytest.fixture(scope="module")
def trainable_info_keys():
    """Keys returned on any call to a subclass of `ray.tune.Trainable`."""
    return {
        "experiment_id",
        "date",
        "timestamp",
        "time_this_iter_s",
        "time_total_s",
        "pid",
        "hostname",
        "node_ip",
        "config",
        "time_since_restore",
        "timesteps_since_restore",
        "iterations_since_restore",
    }


@pytest.fixture
def model_update_interval():
    return 1


@pytest.fixture
def policy_improvement_interval():
    return 1


@pytest.fixture
def policy_improvements():
    return 1


@pytest.fixture
def train_batch_size():
    return 32


@pytest.fixture
def rollout_fragment_length():
    return 10


@pytest.fixture
def learning_starts(rollout_fragment_length):
    return rollout_fragment_length


@pytest.fixture
def num_workers():
    return 0


@pytest.fixture
def buffer_size():
    return 100


@pytest.fixture
def timesteps_per_iteration(rollout_fragment_length):
    return rollout_fragment_length


@pytest.fixture
def evaluation_interval():
    return 1


@pytest.fixture(scope="module")
def dummy_policy_cls():
    class DummyPolicy(Policy):
        # pylint:disable=abstract-method,too-many-arguments
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.param_seq = itertools.count()
            self.param = next(self.param_seq)
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


@pytest.fixture(scope="module", params="MockEnv Navigation".split())
def env_name(request):
    return request.param


@pytest.fixture(scope="module")
def env_(env_name):
    from raylab.envs import get_env_creator

    return get_env_creator(env_name)({})


@pytest.fixture(scope="module")
def policy_fn(env_, env_name):
    def make_policy(policy_cls, config):
        config["env"] = env_name
        return policy_cls(env_.observation_space, env_.action_space, config)

    return make_policy


@pytest.fixture
def env_samples(env_):
    return fake_batch(env_.observation_space, env_.action_space, batch_size=10)
