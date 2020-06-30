# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import functools
import random

import numpy as np
import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy import ModelSamplingMixin
from raylab.policy import TorchPolicy
from raylab.utils.debug import fake_batch

ENSEMBLE_SIZE = (1, 4)
ROLLOUT_SCHEDULE = ([(0, 1), (200, 10)], [(7, 2)])


class DummyPolicy(ModelSamplingMixin, TorchPolicy):
    # pylint:disable=all
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def reward_fn(obs, act, new_obs):
            return act.norm(p=1, dim=-1)

        def termination_fn(obs, act, new_obs):
            return torch.randn(obs.shape[:-1]) > 0

        self.reward_fn = reward_fn
        self.termination_fn = termination_fn

    @staticmethod
    def get_default_config():
        return {
            "model_sampling": ModelSamplingMixin.model_sampling_defaults(),
            "module": {"type": "ModelBasedSAC"},
            "seed": None,
        }


@pytest.fixture(scope="module")
def policy_cls(obs_space, action_space):
    return functools.partial(DummyPolicy, obs_space, action_space)


@pytest.fixture(
    scope="module", params=ENSEMBLE_SIZE, ids=(f"Ensemble({s})" for s in ENSEMBLE_SIZE)
)
def ensemble_size(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=ROLLOUT_SCHEDULE,
    ids=(f"Rollout({s})" for s in ROLLOUT_SCHEDULE),
)
def rollout_schedule(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size, rollout_schedule):
    return {
        "model_sampling": {
            "num_elites": (ensemble_size + 1) // 2,
            "rollout_schedule": rollout_schedule,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
        "seed": 123,
    }


@pytest.fixture(scope="module")
def policy(policy_cls, config):
    return policy_cls(config)


def test_init(policy):
    config = policy.config["model_sampling"]

    assert hasattr(policy, "rng")

    assert hasattr(policy, "elite_models")
    assert isinstance(policy.elite_models, list)
    assert len(policy.elite_models) == config["num_elites"]
    assert all(e is m for e, m in zip(policy.elite_models, policy.module.models))


@pytest.fixture(
    params=(lambda: float("nan"), lambda: random.random()),
    ids="NaNLosses RandLosses".split(),
)
def losses(request, ensemble_size):
    return [request.param() for _ in range(ensemble_size)]


def test_setup_sampling_models(policy, losses):
    policy.setup_sampling_models(losses)

    expected_elites = [policy.module.models[i] for i in np.argsort(losses)]
    assert all(ee is em for ee, em in zip(expected_elites, policy.elite_models))


def test_generate_virtual_sample_batch(policy, rollout_schedule):
    obs_space, action_space = policy.observation_space, policy.action_space
    initial_states = 10
    samples = fake_batch(obs_space, action_space, batch_size=initial_states)
    batch = policy.generate_virtual_sample_batch(samples)

    assert isinstance(batch, SampleBatch)
    assert SampleBatch.CUR_OBS in batch
    assert SampleBatch.ACTIONS in batch
    assert SampleBatch.NEXT_OBS in batch
    assert SampleBatch.REWARDS in batch
    assert SampleBatch.DONES in batch

    policy.global_timestep = 10
    for timestep, value in rollout_schedule:
        if policy.global_timestep >= timestep:
            break
    min_length = value

    min_count = min_length * initial_states
    assert batch.count >= min_count
    assert batch[SampleBatch.CUR_OBS].shape == (batch.count,) + obs_space.shape
    assert batch[SampleBatch.ACTIONS].shape == (batch.count,) + action_space.shape
    assert batch[SampleBatch.NEXT_OBS].shape == (batch.count,) + obs_space.shape
    assert batch[SampleBatch.REWARDS].shape == (batch.count,)
    assert batch[SampleBatch.REWARDS].shape == (batch.count,)
