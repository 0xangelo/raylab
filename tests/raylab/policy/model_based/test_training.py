import functools
import math

import pytest
import torch
from ray.rllib import SampleBatch

from raylab.options import configure
from raylab.options import option
from raylab.policy import ModelTrainingMixin
from raylab.policy import OptimizerCollection
from raylab.policy.model_based.training import Evaluator
from raylab.torch.optim import build_optimizer
from raylab.utils.debug import fake_batch


class DummyLoss:
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, ensemble_size):
        self.ensemble_size = ensemble_size

    def _losses(self):
        return torch.randn(self.ensemble_size)

    def __call__(self, _):
        self.last_losses = losses = self._losses()
        return losses.requires_grad_(True).mean(), {
            "loss(models)": losses.mean().item()
        }


@pytest.fixture
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


@pytest.fixture(scope="module")
def policy_cls(base_policy_cls):
    @configure
    @option("model_training", ModelTrainingMixin.model_training_defaults())
    @option("model_warmup", ModelTrainingMixin.model_training_defaults())
    @option("module/type", "ModelBasedSAC")
    class Policy(ModelTrainingMixin, base_policy_cls):
        # pylint:disable=abstract-method
        def __init__(self, config):
            super().__init__(config)
            self.loss_train = DummyLoss(len(self.module.models))
            self.loss_warmup = DummyLoss(len(self.module.models))

        @property
        def model_training_loss(self):
            return self.loss_train

        @property
        def model_warmup_loss(self):
            return self.loss_warmup

        def _make_optimizers(self):
            optimizers = super()._make_optimizers()
            optimizers["models"] = build_optimizer(self.module.models, {"type": "Adam"})
            return optimizers

    return Policy


@pytest.fixture(scope="module", params=(1, 4), ids=lambda s: f"Ensemble({s})")
def ensemble_size(request):
    return request.param


@pytest.fixture
def max_epochs():
    return 5


@pytest.fixture
def config(ensemble_size, max_epochs):
    options = {
        "model_training": {
            "dataloader": {"batch_size": 32, "replacement": False},
            "max_epochs": max_epochs,
            "max_grad_steps": None,
            "max_time": None,
            "improvement_threshold": 0,
            "patience_epochs": None,
            "holdout_ratio": 0.2,
        },
        "model_warmup": {
            "dataloader": {"batch_size": 64, "replacement": True},
            "max_epochs": max_epochs * 2,
            "max_grad_steps": None,
            "max_time": None,
            "improvement_threshold": None,
            "holdout_ratio": 0.2,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
    }
    return {"policy": options}


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(config)


def test_optimize_model(policy, mocker, samples, max_epochs):
    init = mocker.spy(Evaluator, "__init__")
    # train_loss = mocker.spy(DummyLoss, "__call__")
    # _train_model_epochs = mocker.spy(ModelTrainingMixin, "_train_model_epochs")

    losses, info = policy.optimize_model(samples, warmup=False)

    assert init.called
    # assert policy.loss_train is train_loss.call_args.args[1][0]
    # assert policy.model_training_spec is _train_model_epochs.call_args.kwargs["spec"]

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] == max_epochs
    assert "train/loss(models)" in info
    assert "eval/loss(models)" in info
    assert "grad_norm(models)" in info


def test_warmup_model(policy, mocker, samples, max_epochs):
    init = mocker.spy(Evaluator, "__init__")
    # warmup_loss = mocker.spy(DummyLoss, "__call__")
    # _train_model_epochs = mocker.spy(ModelTrainingMixin, "_train_model_epochs")

    losses, info = policy.optimize_model(samples, warmup=True)

    assert not init.called
    # assert policy.loss_warmup is warmup_loss.call_args.args[1][0]
    # assert policy.model_warmup_spec is _train_model_epochs.call_args.kwargs["spec"]

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] == max_epochs * 2
    assert "train/loss(models)" in info
    assert "eval/loss(models)" not in info
    assert "grad_norm(models)" in info


def test_optimize_with_no_eval(policy, mocker, samples, max_epochs):
    init = mocker.spy(Evaluator, "__init__")
    policy.model_training_spec.holdout_ratio = 0.0
    losses, info = policy.optimize_model(samples, warmup=False)
    assert not init.called

    assert isinstance(losses, list)
    assert all(math.isnan(loss) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] == max_epochs
    assert "train/loss(models)" in info
    assert "eval/loss(models)" not in info
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
        "holdout_ratio": 0.2,
        "max_holdout": None,
    }
    config["policy"]["model_training"].update(model_training)
    return policy_cls(config)


class WorseningLoss(DummyLoss):
    def _losses(self):
        return torch.ones(self.ensemble_size)


def test_early_stop(patient_policy, patience_epochs, ensemble_size, samples):
    patient_policy.loss_train = patient_policy.loss_warmup = WorseningLoss(
        ensemble_size
    )

    _, info = patient_policy.optimize_model(samples, warmup=False)

    assert info["model_epochs"] == patience_epochs
