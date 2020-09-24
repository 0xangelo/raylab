# pylint:disable=missing-module-docstring
import copy
import statistics as stats
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from raylab.options import option
from raylab.policy.losses import Loss
from raylab.policy.modules.model import SME
from raylab.torch.utils import convert_to_tensor
from raylab.utils.lightning import supress_stderr
from raylab.utils.lightning import supress_stdout
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict


# ======================================================================================
# LightningModel
# ======================================================================================


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors,arguments-differ,missing-docstring
    def __init__(self, model: nn.Module, loss: Loss, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.configure_losses(loss)
        self.optimizer = optimizer

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer

    def configure_losses(self, loss: Loss):
        self.train_loss = self.val_loss = self.test_loss = loss

    def forward(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        return self.train_loss(batch)

    def training_step(self, batch: TensorDict, _) -> pl.TrainResult:
        loss, info = self.train_loss(batch)
        info = self.stat_to_tensor_dict(info)
        result = pl.TrainResult(loss, early_stop_on=loss)
        result.log("train/loss", loss)
        result.log_dict({"train/" + k: v for k, v in info.items()})
        return result

    def validation_step(self, batch: TensorDict, _) -> pl.EvalResult:
        loss, info = self.val_loss(batch)
        info = self.stat_to_tensor_dict(info)
        result = pl.EvalResult(early_stop_on=loss)
        result.log("val/loss", loss)
        result.log_dict({"val/" + k: v for k, v in info.items()})
        return result

    def test_step(self, batch: TensorDict, _) -> pl.EvalResult:
        loss, info = self.test_loss(batch)
        info = self.stat_to_tensor_dict(info)
        result = pl.EvalResult()
        result.log("test/loss", loss)
        result.log_dict({"test/" + k: v for k, v in info.items()})
        return result

    def stat_to_tensor_dict(self, info: StatDict) -> TensorDict:
        return {k: convert_to_tensor(v, self.device) for k, v in info.items()}


# ======================================================================================
# DataModule
# ======================================================================================


@dataclass
class DatamoduleSpec(DataClassJsonMixin):
    """Specifications for creating the data module.

    Attributes:
        holdout_ratio: Fraction of replay buffer to use as validation dataset
        max_holdout: Maximum number of samples to use as validation dataset
        batch_size: Size of minibatch for dynamics model training
        shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``True``).
        num_workers: How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
    """

    holdout_ratio: float = 0.2
    max_holdout: Optional[int] = None
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0  # Use at least one worker for speedup

    def __post_init__(self):
        assert self.holdout_ratio < 1.0, "Holdout data cannot be the entire dataset"
        assert (
            not self.max_holdout or self.max_holdout >= 0
        ), "Maximum number of holdout samples must be non-negative"
        assert self.batch_size > 0, "Model batch size must be positive"


class DataModule(pl.LightningDataModule):
    """Data module from experience replay buffer

    Args:
        replay: Experience replay buffer
        spec: Data loading especifications
    """

    # pylint:disable=abstract-method
    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, replay: NumpyReplayBuffer, spec: DatamoduleSpec):
        assert isinstance(replay, NumpyReplayBuffer)
        super().__init__()
        self.replay_dataset = ReplayDataset(replay)
        self.spec = spec

    def setup(self, stage=None):
        dataset = self.replay_dataset
        spec = self.spec
        replay_count = len(dataset)
        max_holdout = spec.max_holdout or replay_count
        val_size = min(round(replay_count * spec.holdout_ratio), max_holdout)
        self.train_dataset, self.val_dataset = random_split(
            dataset, (replay_count - val_size, val_size)
        )

    def train_dataloader(self, *args, **kwargs):
        spec = self.spec
        kwargs = dict(
            shuffle=spec.shuffle,
            batch_size=spec.batch_size,
            num_workers=spec.num_workers,
        )
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self, *args, **kwargs):
        if len(self.val_dataset) == 0:
            return None

        spec = self.spec
        kwargs = dict(
            shuffle=False, batch_size=spec.batch_size, num_workers=spec.num_workers
        )
        return DataLoader(self.val_dataset, **kwargs)


class ReplayDataset(Dataset):
    """Adapter for using a replay buffer as an map-style dataset."""

    def __init__(self, replay: NumpyReplayBuffer):
        self.replay = replay

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx: int):
        return self.replay[idx]


# ======================================================================================
# EarlyStopping
# ======================================================================================


class EarlyStopping(pl.callbacks.EarlyStopping):
    # pylint:disable=missing-docstring
    _train_outputs: List[Tuple[Tensor, StatDict]]
    _val_outputs: List[Tuple[Tensor, StatDict]]
    _loss: Tuple[List[float], StatDict] = None
    _module_state: Optional[dict] = None

    def __warn_deprecated_monitor_key(self):
        pass  # Disable annoying UserWarning

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_outputs = []
        super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # pylint:disable=too-many-arguments
        self._train_outputs += [pl_module.val_loss.last_output]
        super().on_train_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_outputs = []
        super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        # pylint:disable=too-many-arguments
        self._val_outputs += [pl_module.val_loss.last_output]
        super().on_validation_batch_end(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )

    def _run_early_stopping_check(self, trainer, pl_module):
        if self.patience is None:
            # Always save latest outputs
            self.save_outputs()
        else:
            super()._run_early_stopping_check(trainer, pl_module)
            # Save outputs only if improved or none have been logged yet
            if self.wait_count == 0:  # Improved
                self.save_outputs()
                self.save_module_state(pl_module)
            elif self._loss is None:
                self.save_outputs()

    def save_outputs(self):
        if self.monitor == "val_early_stop_on":
            epoch_outputs = self._val_outputs
        else:
            epoch_outputs = self._train_outputs

        epoch_losses, epoch_infos = zip(*epoch_outputs)
        model_losses = torch.stack(epoch_losses, dim=0).mean(dim=0).tolist()
        model_infos = {k: stats.mean(i[k] for i in epoch_infos) for k in epoch_infos[0]}
        self._loss = (model_losses, model_infos)

    def save_module_state(self, pl_module):
        self._module_state = copy.deepcopy(pl_module.state_dict())

    def state_dict(self):
        state = super().state_dict()
        state.update(loss=self._loss, module=self._module_state)
        return state

    def load_state_dict(self, state_dict):
        self._loss = state_dict["loss"]
        self._module_state = copy.deepcopy(state_dict["module"])
        used = set("loss module".split())
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in used})


# ======================================================================================
# Model Trainer
# ======================================================================================


@dataclass
class LightningTrainerSpec(DataClassJsonMixin):
    """Specifications for Lightning trainers.

    Attributes:
        max_epochs: Maximum number of full model passes through the data
        max_steps: Maximum number of model gradient steps
        patience: Tolerate this many epochs of successive performance
            degradation. If None, disables early stopping.
        improvement_delta: Minimum expected absolute improvement in model
            validation loss
    """

    max_epochs: Optional[int] = 1
    max_steps: Optional[int] = None
    patience: Optional[int] = 1
    improvement_delta: float = 0.0

    def __post_init__(self):
        if self.max_epochs is None:
            self.max_epochs = self.max_steps

        assert self.max_epochs > 0, "Maximum number of epochs must be positive"
        assert not self.max_steps or self.max_steps > 0
        assert (
            self.patience is None or self.patience >= 0
        ), "Patience must be nonnegative or None"
        assert isinstance(
            self.improvement_delta, float
        ), "Improvement threshold must be a scalar"

    def build_trainer(self, check_val: bool) -> pl.Trainer:
        """Returns the Pytorch Lightning configured with this spec."""
        early_stopping = EarlyStopping(
            min_delta=self.improvement_delta,
            patience=self.patience,
            mode="min",
            strict=False,
        )
        return pl.Trainer(
            logger=False,
            num_sanity_val_steps=2 if check_val else 0,
            early_stop_callback=early_stopping,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            progress_bar_refresh_rate=0,
            track_grad_norm=2,
            # gradient_clip_val=1e4,  # Broken
        )


@dataclass
class TrainingSpec(DataClassJsonMixin):
    """Specifications for training the model.

    Attributes:
        datamodule: Specifications for creating the data module
        training: Specifications for model training
        warmup: Specifications for model warmup
    """

    datamodule: DatamoduleSpec = field(default_factory=DatamoduleSpec)
    training: LightningTrainerSpec = field(default_factory=LightningTrainerSpec)
    warmup: LightningTrainerSpec = field(default_factory=LightningTrainerSpec)


class LightningModelTrainer:
    """Model training behavior for TorchPolicy instances via PyTorch Lightning.

    Args:
        models: Stochastic model ensemble
        loss_fn: Loss associated with the model ensemble
        optimizer: Optimizer associated with the model ensemble
        replay: Experience replay buffer
        config: Dictionary containg `model_training` and `model_warmup` dicts

    Attributes:
        pl_model: Pytorch Lightning model
        datamodule: Lightning data module
        spec: Specifications for training the model
        training_loss: Loss function used for model training and evaluation
        warmup_loss: Loss function used for model warm-up.
    """

    pl_model: LightningModel
    datamodule: DataModule
    spec: TrainingSpec

    def __init__(
        self,
        models: SME,
        loss_fn: Loss,
        optimizer: Optimizer,
        replay: NumpyReplayBuffer,
        config: dict,
    ):
        # pylint:disable=too-many-arguments
        self.spec = TrainingSpec.from_dict(config["model_training"])
        self.pl_model = LightningModel(model=models, loss=loss_fn, optimizer=optimizer)
        self.datamodule = DataModule(replay, self.spec.datamodule)
        self.training_loss = self.warmup_loss = loss_fn

    def optimize(self, warmup: bool = False) -> Tuple[List[float], StatDict]:
        """Update models using replay buffer data.

        Args:
            warmup: Whether to train with warm-up loss and spec

        Returns:
            A tuple with a list of each model's evaluation loss and a dictionary
            with training statistics
        """
        loss_fn = self.warmup_loss if warmup else self.training_loss
        self.pl_model.configure_losses(loss_fn)

        trainer_spec = self.spec.warmup if warmup else self.spec.training
        trainer = trainer_spec.build_trainer(check_val=warmup)

        losses, info = self.run_training(
            model=self.pl_model, trainer=trainer, datamodule=self.datamodule
        )
        return losses, info

    @staticmethod
    @supress_stderr
    @supress_stdout
    def run_training(
        model: LightningModel, trainer: pl.Trainer, datamodule: DataModule
    ) -> Tuple[List[float], StatDict]:
        """Trains model and handles checkpointing."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pytorch_lightning*")
            trainer.fit(model, datamodule=datamodule)

        saved_state = trainer.early_stop_callback.state_dict()
        losses, info = saved_state["loss"]
        if saved_state["module"]:
            model.load_state_dict(saved_state["module"])
        info.update(
            model_epochs=trainer.current_epoch + 1, model_steps=trainer.global_step
        )
        return losses, info

    @staticmethod
    def add_options(cls_: type) -> type:
        """Add options for classes that may use this class."""
        return option(
            "model_training",
            default=TrainingSpec().to_dict(),
            help=TrainingSpec.__doc__,
        )(cls_)
