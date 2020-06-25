# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import functools
import math

import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy import ModelTrainingMixin
from raylab.policy import OptimizerCollection
from raylab.policy import TorchPolicy
from raylab.policy.model_based.training_mixin import Evaluator
from raylab.pytorch.optim import build_optimizer
from raylab.utils.debug import fake_batch


class DummyLoss:
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)
    ensemble_size: int = 1

    def __call__(self, _):
        losses = torch.randn(self.ensemble_size).requires_grad_(True)
        return losses, {"loss(models)": losses.mean().item()}


class DummyPolicy(ModelTrainingMixin, TorchPolicy):
    # pylint:disable=abstract-method
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        loss = DummyLoss()
        loss.ensemble_size = len(self.module.models)
        self.loss_model = loss

    @staticmethod
    def get_default_config():
        return {
            "model_training": ModelTrainingMixin.model_training_defaults(),
            "module": {"type": "ModelBasedSAC"},
        }

    def make_optimizers(self):
        return {"models": build_optimizer(self.module.models, {"type": "Adam"})}


@pytest.fixture
def train_samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=80)


@pytest.fixture
def eval_samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=20)


@pytest.fixture(scope="module")
def policy_cls(obs_space, action_space):
    return functools.partial(DummyPolicy, obs_space, action_space)


@pytest.fixture(scope="module", params=(1, 4), ids=lambda s: f"Ensemble({s})")
def ensemble_size(request):
    return request.param


@pytest.fixture
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


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(config)


def test_optimize_model(policy, mocker, train_samples, eval_samples):
    init = mocker.spy(Evaluator, "__init__")
    losses, info = policy.optimize_model(train_samples, eval_samples)
    assert init.called

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] >= 0
    assert "train_loss(models)" in info
    assert "eval_loss(models)" in info
    assert "grad_norm(models)" in info


def test_optimize_with_no_eval(policy, mocker, train_samples):
    init = mocker.spy(Evaluator, "__init__")
    losses, info = policy.optimize_model(train_samples)
    assert not init.called

    assert isinstance(losses, list)
    assert all(math.isnan(loss) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] >= 0
    assert "train_loss(models)" in info
    assert "eval_loss(models)" not in info
    assert "grad_norm(models)" in info


@pytest.fixture(
    params=(pytest.param(0, marks=pytest.mark.xfail), 1, 4),
    ids=lambda x: f"Patience({x})",
)
def patience_epochs(request):
    return request.param


@pytest.fixture
def patient_policy(patience_epochs, policy_cls, config):
    model_training = {
        "max_epochs": patience_epochs + 1,
        "max_grad_steps": None,
        "max_time": None,
        "improvement_threshold": 0,
        "patience_epochs": patience_epochs,
    }
    config["model_training"].update(model_training)
    return policy_cls(config)


def test_early_stop(
    patient_policy, patience_epochs, ensemble_size, mocker, train_samples, eval_samples
):
    mock = mocker.patch.object(DummyLoss, "__call__")
    mock.side_effect = lambda _: (torch.ones(ensemble_size).requires_grad_(), {})

    _, info = patient_policy.optimize_model(train_samples, eval_samples)

    assert mock.called

    assert info["model_epochs"] == patience_epochs
