# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import functools
import math

import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy import ModelTrainingMixin
from raylab.policy import TorchPolicy
from raylab.pytorch.optim import build_optimizer
from raylab.utils.debug import fake_batch

ENSEMBLE_SIZE = (1, 4)


class DummyLoss:
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, models):
        self.models = models

    def __call__(self, batch):
        inputs = tuple(batch[k] for k in self.batch_keys)
        losses = torch.stack([-m.log_prob(*inputs).mean() for m in self.models])
        return losses, {"loss(models)": losses.mean().item()}


class DummyOptimizer:
    # pylint:disable=all
    def __init__(self, models):
        self.models = build_optimizer(models, {"type": "Adam"})


class DummyPolicy(ModelTrainingMixin, TorchPolicy):
    # pylint:disable=abstract-method
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_model = DummyLoss(self.module.models)

    @staticmethod
    def get_default_config():
        return {
            "model_training": ModelTrainingMixin.model_training_defaults(),
            "module": {"type": "ModelBasedSAC"},
        }

    def make_optimizer(self):
        return DummyOptimizer(self.module.models)


@pytest.fixture(scope="module")
def policy_cls(obs_space, action_space):
    return functools.partial(DummyPolicy, obs_space, action_space)


@pytest.fixture(
    scope="module", params=ENSEMBLE_SIZE, ids=(f"Ensemble({s})" for s in ENSEMBLE_SIZE)
)
def ensemble_size(request):
    return request.param


@pytest.fixture(scope="module")
def config(ensemble_size):
    return {
        "model_training": {
            "dataloader": {"batch_size": 32, "replacement": False},
            "max_epochs": 10,
            "max_time": 4,
            "improvement_threshold": 0.01,
            "patience_epochs": 5,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
    }


@pytest.fixture(scope="module")
def policy(policy_cls, config):
    return policy_cls(config)


def test_optimize_model(policy):
    obs_space, action_space = policy.observation_space, policy.action_space
    train_samples = fake_batch(obs_space, action_space, batch_size=80)
    eval_samples = fake_batch(obs_space, action_space, batch_size=20)

    losses, info = policy.optimize_model(train_samples, eval_samples)

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] >= 0
    assert "train_loss(models)" in info
    assert "eval_loss(models)" in info
    assert "grad_norm(models)" in info


def test_optimize_with_no_eval(policy):
    obs_space, action_space = policy.observation_space, policy.action_space
    train_samples = fake_batch(obs_space, action_space, batch_size=80)

    losses, info = policy.optimize_model(train_samples)

    print(losses)
    assert isinstance(losses, list)
    assert all(math.isnan(loss) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] >= 0
    assert "train_loss(models)" in info
    assert "eval_loss(models)" not in info
    assert "grad_norm(models)" in info
