# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import collections

import pytest
from ray.rllib import SampleBatch
from ray.rllib.evaluation.metrics import get_learner_stats

from raylab.losses import ModelEnsembleMLE
from raylab.policy import ModelBasedMixin
from raylab.policy import TorchPolicy
from raylab.policy.model_based_mixin import ModelBasedSpec
from raylab.pytorch.optim import build_optimizer
from raylab.utils.debug import fake_batch

ENSEMBLE_SIZE = (1, 4)


class DummyPolicy(ModelBasedMixin, TorchPolicy):
    # pylint:disable=abstract-method
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_model = ModelEnsembleMLE(self.module.models)

    @staticmethod
    def get_default_config():
        return {
            "env": None,
            "env_config": {},
            "seed": None,
            "model_based": ModelBasedSpec().to_dict(),
            "module": {"type": "ModelBasedSAC"},
        }

    def make_optimizer(self):
        optim = build_optimizer(self.module.models, {"type": "Adam"})
        return collections.namedtuple("OptimizerCollection", "models")(models=optim)


@pytest.fixture(scope="module", params=("Navigation", "MockEnv"))
def env_name(request, envs):
    # pylint:disable=unused-argument
    return request.param


@pytest.fixture(scope="module")
def policy_cls(obs_space, action_space, env_name):
    def policy_maker(config):
        config["env"] = env_name
        return DummyPolicy(obs_space, action_space, config)

    return policy_maker


@pytest.fixture(
    scope="module", params=ENSEMBLE_SIZE, ids=(f"Ensemble({s})" for s in ENSEMBLE_SIZE)
)
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size):
    return {
        "model_based": {
            "training": {
                "dataloader": {"batch_size": 32, "replacement": False},
                "max_epochs": 10,
                "max_time": 4,
                "improvement_threshold": 0.01,
                "patience_epochs": 5,
            },
            "rollout_length": 10,
            "num_elites": 1,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
    }


@pytest.fixture(scope="module")
def policy(policy_cls, config):
    return policy_cls(config)


def test_generate_virtual_sample_batch(policy):
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

    total_count = policy.model_based_spec.rollout_length * initial_states
    assert batch.count == total_count
    assert batch[SampleBatch.CUR_OBS].shape == (total_count,) + obs_space.shape
    assert batch[SampleBatch.ACTIONS].shape == (total_count,) + action_space.shape
    assert batch[SampleBatch.NEXT_OBS].shape == (total_count,) + obs_space.shape
    assert batch[SampleBatch.REWARDS].shape == (total_count,)
    assert batch[SampleBatch.REWARDS].shape == (total_count,)


def test_optimize_model(policy):
    obs_space, action_space = policy.observation_space, policy.action_space
    train_samples = fake_batch(obs_space, action_space, batch_size=80)
    eval_samples = fake_batch(obs_space, action_space, batch_size=20)

    info = get_learner_stats(policy.optimize_model(train_samples, eval_samples))

    assert "model_epochs" in info
    assert "loss(models)" in info
    assert all(
        f"loss(models[{i}])" in info
        for i in range(policy.config["module"]["model"]["ensemble_size"])
    )
    assert "loss(models[elites])" in info
    assert "grad_norm(models)" in info
