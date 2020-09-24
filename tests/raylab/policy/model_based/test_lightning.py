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
from raylab.policy.model_based.lightning import DataModule
from raylab.policy.model_based.lightning import LightningModel
from raylab.policy.model_based.lightning import LightningModelTrainer
from raylab.policy.model_based.lightning import TrainingSpec
from raylab.policy.modules import get_module
from raylab.policy.off_policy import off_policy_options
from raylab.policy.off_policy import OffPolicyMixin
from raylab.torch.optim import build_optimizer
from raylab.torch.utils import convert_to_tensor
from raylab.utils.debug import fake_batch
from raylab.utils.replay_buffer import NumpyReplayBuffer


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


@pytest.fixture(scope="module", params=(1, 4), ids=lambda s: f"Ensemble({s})")
def ensemble_size(request):
    return request.param


@pytest.fixture
def models(obs_space, action_space, ensemble_size):
    cnf = {"type": "ModelBasedSAC", "model": {"ensemble_size": ensemble_size}}
    module = get_module(obs_space, action_space, cnf)
    return module.models


@pytest.fixture
def optimizer(models):
    return build_optimizer(models, {"type": "Adam"})


MAXS = ((1, None), (None, 10))


@pytest.fixture(params=(MAXS), ids=lambda x: f"MaxEpochs:{x[0]}-MaxSteps:{x[1]}")
def maxs(request):
    max_epochs, max_steps = request.param
    return dict(max_epochs=max_epochs, max_steps=max_steps)


@pytest.fixture
def max_epochs(maxs):
    return maxs["max_epochs"]


@pytest.fixture
def max_steps(maxs):
    return maxs["max_steps"]


@pytest.fixture
def improvement_delta():
    return 0.0


VALS = ((None, 0.2), (3, 0.0), (3, 0.2))


@pytest.fixture(params=VALS, ids=lambda x: f"Patience:{x[0]}-Holdout%:{x[1]}")
def vals(request):
    patience, holdout_ratio = request.param
    return dict(patience=patience, holdout_ratio=holdout_ratio)


@pytest.fixture
def patience(vals):
    return vals["patience"]


@pytest.fixture
def holdout_ratio(vals):
    return vals["holdout_ratio"]


@pytest.fixture
def config(max_epochs, max_steps, improvement_delta, patience, holdout_ratio):
    trainer_cfg = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "improvement_delta": improvement_delta,
        "patience": patience,
    }
    return {
        "model_training": {
            "datamodule": {
                "batch_size": 32,
                "shuffle": True,
                "holdout_ratio": holdout_ratio,
            },
            "training": trainer_cfg,
            "warmup": trainer_cfg,
        },
    }


@pytest.fixture(scope="module")
def samples(obs_space, action_space):
    return fake_batch(obs_space, action_space, batch_size=256)


@pytest.fixture(scope="module")
def replay(obs_space, action_space, samples):
    replay = NumpyReplayBuffer(obs_space, action_space, size=samples.count)
    replay.add(samples)
    return replay


@pytest.fixture
def build_trainer(models, optimizer, replay, config):
    def builder(model_loss):
        loss_fn = model_loss(models)
        return LightningModelTrainer(models, loss_fn, optimizer, replay, config)

    return builder


def test_init(
    mocker,
    build_trainer,
    max_epochs,
    max_steps,
    improvement_delta,
    patience,
    holdout_ratio,
):
    # pylint:disable=too-many-arguments
    pl_module = mocker.spy(pl.LightningModule, "__init__")
    datamodule = mocker.spy(DataModule, "__init__")

    trainer = build_trainer(DummyLoss)

    assert pl_module.called
    assert datamodule.called

    spec = trainer.spec
    for subspec in (spec.training, spec.warmup):
        assert subspec.max_epochs == (max_epochs or max_steps)
        assert subspec.max_steps == max_steps
        assert subspec.improvement_delta == improvement_delta
        assert subspec.patience == (patience or subspec.max_epochs)
    assert spec.datamodule.holdout_ratio == holdout_ratio


@pytest.fixture
def trainer(build_trainer):
    return build_trainer(DummyLoss)


def test_training(mocker, trainer):
    _test_optimize(mocker, trainer, warmup=False)


def test_warmup(mocker, trainer):
    _test_optimize(mocker, trainer, warmup=True)


def _test_optimize(mocker, trainer: LightningModelTrainer, warmup: bool):
    early_stop = mocker.spy(pl.callbacks.EarlyStopping, "__init__")
    trainer_init = mocker.spy(pl.Trainer, "__init__")
    trainer_fit = mocker.spy(pl.Trainer, "fit")
    trainer_test = mocker.spy(pl.Trainer, "test")
    data_setup = mocker.spy(DataModule, "setup")

    stderr, stdout = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
        losses, info = trainer.optimize(warmup=warmup)

    assert not stderr.getvalue()
    assert not stdout.getvalue()

    assert early_stop.called
    assert trainer_init.called
    assert trainer_fit.called
    assert not trainer_test.called
    assert data_setup.called

    assert isinstance(losses, list)
    assert all(isinstance(loss, float) for loss in losses)

    assert isinstance(info, dict)
    assert "model_epochs" in info
    spec = trainer.spec.warmup if warmup else trainer.spec.training
    assert info["model_epochs"] <= min(spec.max_epochs, spec.patience)
    if spec.max_steps:
        assert info["model_steps"] <= spec.max_steps
    else:
        assert info["model_steps"] > 0


def test_model(trainer, models):
    model = trainer.pl_model
    model_params = set(model.parameters())

    assert isinstance(model, LightningModel)
    mods_params = set(models.parameters())
    assert not set.symmetric_difference(model_params, mods_params)

    optim = model.configure_optimizers()
    optim_params = set(p for g in optim.param_groups for p in g["params"])
    assert not set.symmetric_difference(model_params, optim_params)


def test_test(trainer: LightningModelTrainer, holdout_ratio):
    spec: TrainingSpec = trainer.spec
    pl_model: LightningModel = trainer.pl_model
    datamodule: DataModule = trainer.datamodule

    pl_trainer = spec.training.build_trainer(check_val=False)
    assert isinstance(pl_trainer, pl.Trainer)
    datamodule.setup(None)

    dataloader = (
        datamodule.val_dataloader() if holdout_ratio else datamodule.train_dataloader()
    )
    # with warnings.catch_warnings():
    #     # warnings.filterwarnings("ignore", module="pytorch_lightning*")
    outputs = pl_trainer.test(pl_model, test_dataloaders=dataloader)

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
        self._increasing_seq = itertools.count()
        self.models = models

    def set_grads(self):
        for par in self.models.parameters():
            par.grad = torch.ones_like(par)

    def _losses(self):
        return torch.full(
            (self.ensemble_size,),
            fill_value=float(next(self._increasing_seq)),
            requires_grad=True,
        )

    def __call__(self, *args, **kwargs):
        self.set_grads()
        return super().__call__(*args, **kwargs)


def test_checkpointing(build_trainer):
    trainer = build_trainer(WorseningLoss)
    patience = 2
    spec = trainer.spec.training
    spec.max_epochs = 1000
    spec.max_steps = None
    spec.patience = patience

    pl_model = trainer.pl_model
    datamodule = trainer.datamodule

    pl_trainer = spec.build_trainer(check_val=False)

    before_params = copy.deepcopy(list(pl_model.parameters()))
    losses, info = trainer.run_training(pl_model, pl_trainer, datamodule)
    assert info["model_epochs"] == patience + 1

    after_params = list(pl_model.parameters())
    assert all([torch.allclose(b, a) for b, a in zip(before_params, after_params)])
