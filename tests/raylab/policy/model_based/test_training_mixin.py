import functools
import math

import pytest
import torch
from ray.rllib import SampleBatch

from raylab.agents.options import RaylabOptions
from raylab.policy import ModelTrainingMixin
from raylab.policy import OptimizerCollection
from raylab.policy.model_based.training_mixin import Evaluator
from raylab.pytorch.optim import build_optimizer
from raylab.utils.debug import fake_batch


class DummyLoss:
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, ensemble_size):
        self.ensemble_size = ensemble_size

    def __call__(self, _):
        losses = torch.randn(self.ensemble_size).requires_grad_(True)
        return losses, {"loss(models)": losses.mean().item()}


@pytest.fixture
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


@pytest.fixture(scope="module")
def policy_cls(base_policy_cls):
    class Policy(ModelTrainingMixin, base_policy_cls):
        # pylint:disable=abstract-method
        def __init__(self, config):
            super().__init__(config)
            self.loss_train = DummyLoss(len(self.module.models))
            self.loss_warmup = DummyLoss(len(self.module.models))

        @property
        def options(self):
            options = RaylabOptions()
            options.set_option(
                "model_training", ModelTrainingMixin.model_training_defaults()
            )
            options.set_option(
                "model_warmup", ModelTrainingMixin.model_training_defaults()
            )
            options.set_option("model/type", "ModelBasedSAC")
            return options

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
def config(ensemble_size):
    return {
        "model_training": {
            "dataloader": {"batch_size": 32, "replacement": False},
            "max_epochs": 5,
            "max_grad_steps": None,
            "max_time": None,
            "improvement_threshold": 0,
            "patience_epochs": None,
            "holdout_ratio": 0.2,
        },
        "model_warmup": {
            "dataloader": {"batch_size": 64, "replacement": True},
            "max_epochs": 10,
            "max_grad_steps": None,
            "max_time": None,
            "improvement_threshold": None,
            "holdout_ratio": 0.2,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
    }


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(config)


def test_optimize_model(policy, mocker, samples):
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
    assert info["model_epochs"] == 5
    assert "train/loss(models)" in info
    assert "eval/loss(models)" in info
    assert "grad_norm(models)" in info


def test_warmup_model(policy, mocker, samples):
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
    assert info["model_epochs"] == 10
    assert "train/loss(models)" in info
    assert "eval/loss(models)" not in info
    assert "grad_norm(models)" in info


def test_optimize_with_no_eval(policy, mocker, samples):
    init = mocker.spy(Evaluator, "__init__")
    policy.model_training_spec.holdout_ratio = 0.0
    losses, info = policy.optimize_model(samples, warmup=False)
    assert not init.called

    assert isinstance(losses, list)
    assert all(math.isnan(loss) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] == 5
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
    config["model_training"].update(model_training)
    return policy_cls(config)


def test_early_stop(patient_policy, patience_epochs, ensemble_size, mocker, samples):
    mock = mocker.patch.object(DummyLoss, "__call__")
    mock.side_effect = lambda _: (torch.ones(ensemble_size).requires_grad_(), {})

    _, info = patient_policy.optimize_model(samples, warmup=False)

    assert mock.called
    assert info["model_epochs"] == patience_epochs
