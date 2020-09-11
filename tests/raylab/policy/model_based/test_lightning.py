import functools
import math
import warnings

import pytest
import pytorch_lightning as pl
import torch
from ray.rllib import SampleBatch
from torch.utils.data import DataLoader

from raylab.agents.options import RaylabOptions
from raylab.policy import OptimizerCollection
from raylab.policy.model_based.lightning import LightningModel
from raylab.policy.model_based.lightning import LightningModelMixin
from raylab.torch.optim import build_optimizer
from raylab.utils.debug import fake_batch


class DummyLoss:
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, ensemble_size):
        self.ensemble_size = ensemble_size

    def __call__(self, _):
        losses = torch.randn(self.ensemble_size).requires_grad_(True)
        return losses.sum(), {"loss(models)": losses.mean().item()}


@pytest.fixture(scope="module")
def policy_cls(base_policy_cls):
    class Policy(LightningModelMixin, base_policy_cls):
        # pylint:disable=abstract-method
        def __init__(self, model_loss, config):
            super().__init__(config)
            self.loss_train = model_loss(len(self.module.models))
            self.loss_warmup = model_loss(len(self.module.models))

        @property
        def options(self):
            options = RaylabOptions()
            options.set_option(
                "model_training", LightningModelMixin.model_training_defaults()
            )
            options.set_option(
                "model_warmup", LightningModelMixin.model_training_defaults()
            )
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


@pytest.fixture(params=(1,), ids=lambda x: f"MaxEpochs:{x}")
def max_epochs(request):
    return request.param


@pytest.fixture(params=(None,), ids=lambda x: f"MaxSteps:{x}")
def max_steps(request):
    return request.param


@pytest.fixture
def improvement_delta():
    return 0.0


@pytest.fixture
def patience():
    return 3


@pytest.fixture
def holdout_ratio():
    return 0.2


@pytest.fixture
def config(
    max_epochs, max_steps, improvement_delta, patience, holdout_ratio, ensemble_size
):
    # pylint:disable=too-many-arguments
    return {
        "model_training": {
            "dataloader": {"batch_size": 32, "shuffle": True},
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "improvement_delta": improvement_delta,
            "patience": patience,
            "holdout_ratio": holdout_ratio,
        },
        "model_warmup": {
            "dataloader": {"batch_size": 64, "shuffle": True},
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "improvement_delta": improvement_delta,
            "patience": patience,
            "holdout_ratio": holdout_ratio,
        },
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}},
    }


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(DummyLoss, config)


def test_init(
    policy, max_epochs, max_steps, improvement_delta, patience, holdout_ratio
):
    for spec in (policy.model_training_spec, policy.model_warmup_spec):
        assert spec.max_epochs == max_epochs
        assert spec.max_steps == max_steps
        assert spec.improvement_delta == improvement_delta
        assert spec.patience == patience
        assert spec.holdout_ratio == holdout_ratio


@pytest.fixture
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


def test_optimize_model(policy, mocker, samples):
    _test_optimization(policy, mocker, samples, warmup=False)


def test_warmup_model(policy, mocker, samples):
    _test_optimization(policy, mocker, samples, warmup=True)


def _test_optimization(policy, mocker, samples, warmup):
    pl_module = mocker.spy(pl.LightningModule, "__init__")
    early_stop = mocker.spy(pl.callbacks.EarlyStopping, "__init__")
    trainer_init = mocker.spy(pl.Trainer, "__init__")
    trainer_fit = mocker.spy(pl.Trainer, "fit")
    trainer_test = mocker.spy(pl.Trainer, "test")

    losses, info = policy.optimize_model(samples, warmup=warmup)

    assert pl_module.called
    assert early_stop.called
    assert trainer_init.called
    assert trainer_fit.called
    assert trainer_test.called

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] > 0


def test_model(policy):
    loss_fn = policy.model_training_loss
    model = policy.get_lightning_model(loss_fn)
    model_params = set(model.parameters())

    assert isinstance(model, LightningModel)
    mods_params = set(policy.module.models.parameters())
    assert not set.symmetric_difference(model_params, mods_params)

    optim = model.configure_optimizers()
    optim_params = set(p for g in optim.param_groups for p in g["params"])
    assert not set.symmetric_difference(model_params, optim_params)


def test_trainer_output(policy, samples):
    spec = policy.model_training_spec
    loss_fn = policy.model_training_loss
    train, val = spec.train_val_loaders(
        *spec.train_val_tensors(samples, loss_fn.batch_keys, policy.convert_to_tensor)
    )

    assert isinstance(train, DataLoader)
    assert isinstance(val, DataLoader)

    model = LightningModel(
        model=policy.module.models, loss=loss_fn, optimizer=policy.optimizers["models"]
    )
    trainer = policy.get_trainer(spec)
    assert isinstance(trainer, pl.Trainer)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="pytorch_lightning*")
        outputs = trainer.test(model, test_dataloaders=val)

    assert isinstance(outputs, (list, tuple))
    assert len(outputs) == 1
    info = outputs[0]
    assert isinstance(info, dict)
    assert "test/loss" in info
    assert isinstance(info["test/loss"], float)
