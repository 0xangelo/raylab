import contextlib
import copy
import functools
import io
import itertools
import math
import warnings

import pytest
import pytorch_lightning as pl
import torch
from ray.rllib import SampleBatch
from torch.utils.data import DataLoader

from raylab.options import configure
from raylab.policy import OptimizerCollection
from raylab.policy.losses import Loss
from raylab.policy.model_based.lightning import LightningModel
from raylab.policy.model_based.lightning import LightningModelTrainer
from raylab.torch.optim import build_optimizer
from raylab.torch.utils import convert_to_tensor
from raylab.utils.debug import fake_batch


class DummyLoss(Loss):
    # pylint:disable=all
    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, models):
        self.ensemble_size = len(models)

    def _losses(self):
        return torch.randn(self.ensemble_size)

    def __call__(self, _):
        losses = self._losses().requires_grad_(True)
        info = {"loss(models)": losses.mean().item()}
        self.last_output = (losses, info)
        return losses.mean(), info


@pytest.fixture(scope="module")
def policy_cls(base_policy_cls):
    @configure
    @LightningModelTrainer.add_options
    class Policy(base_policy_cls):
        # pylint:disable=abstract-method
        def __init__(self, model_loss, config):
            super().__init__(config)
            models = self.module.models
            self.model_trainer = LightningModelTrainer(
                models, model_loss(models), self.optimizers["models"], self.config
            )

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
    options = {
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
    return {"policy": options}


def test_init(
    mocker,
    policy_cls,
    config,
    max_epochs,
    max_steps,
    improvement_delta,
    patience,
    holdout_ratio,
):
    # pylint:disable=too-many-arguments
    model_trainer = mocker.spy(LightningModelTrainer, "__init__")
    pl_module = mocker.spy(pl.LightningModule, "__init__")

    policy = policy_cls(DummyLoss, config)

    assert pl_module.called
    assert model_trainer.called

    trainer = policy.model_trainer
    for spec in (trainer.model_training_spec, trainer.model_warmup_spec):
        assert spec.max_epochs == max_epochs
        assert spec.max_steps == max_steps
        assert spec.improvement_delta == improvement_delta
        assert spec.patience == patience
        assert spec.holdout_ratio == holdout_ratio


@pytest.fixture
def policy(policy_cls, config):
    return policy_cls(DummyLoss, config)


@pytest.fixture
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


@pytest.mark.slow
def test_optimize_model(policy, mocker, samples):
    _test_optimization(policy, mocker, samples, warmup=False)


@pytest.mark.slow
def test_warmup_model(policy, mocker, samples):
    _test_optimization(policy, mocker, samples, warmup=True)


def _test_optimization(policy, mocker, samples, warmup):
    early_stop = mocker.spy(pl.callbacks.EarlyStopping, "__init__")
    trainer_init = mocker.spy(pl.Trainer, "__init__")
    trainer_fit = mocker.spy(pl.Trainer, "fit")
    trainer_test = mocker.spy(pl.Trainer, "test")

    stderr, stdout = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
        losses, info = policy.model_trainer.optimize(samples, warmup=warmup)

    assert not stderr.getvalue()
    assert not stdout.getvalue()

    assert early_stop.called
    assert trainer_init.called
    assert trainer_fit.called
    assert not trainer_test.called

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    assert info["model_epochs"] > 0


def test_model(policy):
    model = policy.model_trainer.pl_model
    model_params = set(model.parameters())

    assert isinstance(model, LightningModel)
    mods_params = set(policy.module.models.parameters())
    assert not set.symmetric_difference(model_params, mods_params)

    optim = model.configure_optimizers()
    optim_params = set(p for g in optim.param_groups for p in g["params"])
    assert not set.symmetric_difference(model_params, optim_params)


@pytest.fixture
def model_trainer(policy):
    return policy.model_trainer


@pytest.mark.slow
def test_trainer_output(policy, model_trainer, samples):
    spec = model_trainer.model_training_spec
    loss_fn = model_trainer.model_training_loss
    pl_model = model_trainer.pl_model
    train, val = spec.train_val_loaders(
        *spec.train_val_tensors(
            samples,
            loss_fn.batch_keys,
            lambda x: convert_to_tensor(x, device=pl_model.device),
        )
    )

    assert isinstance(train, DataLoader)
    assert isinstance(val, DataLoader)

    trainer = spec.build_trainer()
    assert isinstance(trainer, pl.Trainer)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="pytorch_lightning*")
        outputs = trainer.test(pl_model, test_dataloaders=val)

    assert isinstance(outputs, (list, tuple))
    assert len(outputs) == 1
    info = outputs[0]
    assert isinstance(info, dict)
    assert "test/loss" in info
    assert isinstance(info["test/loss"], float)
    assert "test/loss(models)" in info


class WorseningLoss(DummyLoss):
    def __init__(self, models):
        super().__init__(models)
        self.models = models
        self._increasing_seq = itertools.count()

    def _losses(self):
        return torch.full(
            (self.ensemble_size,),
            fill_value=float(next(self._increasing_seq)),
            requires_grad=True,
        )

    def _perturb_models(self):
        perturbations = [torch.randn_like(p) * 0.01 for p in self.models.parameters()]
        for par, per in zip(self.models.parameters(), perturbations):
            par.data.add_(per)

    def __call__(self, *args, **kwargs):
        self._perturb_models()
        return super().__call__(*args, **kwargs)


@pytest.fixture
def worsening_policy(policy_cls, config):
    return policy_cls(WorseningLoss, config)


@pytest.mark.slow
def test_checkpointing(worsening_policy, samples):
    trainer = worsening_policy.model_trainer
    patience = 2
    trainer.model_training_spec.max_epochs = 1000
    trainer.model_training_spec.patience = patience

    init_params = copy.deepcopy(list(trainer.pl_model.model.parameters()))
    losses, info = trainer.optimize(samples, warmup=False)
    assert info["model_epochs"] == patience + 1

    after_params = list(trainer.pl_model.model.parameters())
    assert all([torch.allclose(i, p) for i, p in zip(init_params, after_params)])
