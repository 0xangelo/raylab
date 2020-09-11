# pylint:disable=missing-docstring
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from ray.rllib import SampleBatch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from raylab.policy.losses import Loss
from raylab.torch.utils import convert_to_tensor
from raylab.torch.utils import TensorDictDataset
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict
from raylab.utils.lightning import supress_stderr
from raylab.utils.lightning import supress_stdout


@dataclass
class DataloaderSpec(DataClassJsonMixin):
    """Specifications for creating the data loader.

    Attributes:
        batch_size: Size of minibatch for dynamics model training
        shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``True``).
        num_workers: How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
    """

    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 1  # Use at least one worker for speedup

    def __post_init__(self):
        assert self.batch_size > 0, "Model batch size must be positive"

    def build_dataloader(self, tensors: TensorDict, train: bool = True) -> DataLoader:
        """Returns a dataloader for the given tensors based on specifications.

        Args:
            train: Whether the output is intended to be a train dataloader. If
                false, disables shuffling even if `self.shuffle` is true.
        """
        return DataLoader(
            dataset=TensorDictDataset(tensors),
            shuffle=False if not train else self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


@dataclass
class TrainingSpec(DataClassJsonMixin):
    """Specifications for training the model.

    Attributes:
        dataloader: specifications for creating the data loader
        max_epochs: Maximum number of full model passes through the data
        max_steps: Maximum number of model gradient steps
        patience: Number of epochs to wait for validation loss to improve
        improvement_delta: Minimum expected relative improvement in model
            validation loss
        holdout_ratio: Fraction of replay buffer to use as validation dataset
        max_holdout: Maximum number of samples to use as validation dataset
    """

    # pylint:disable=too-many-instance-attributes
    dataloader: DataloaderSpec = field(default_factory=DataloaderSpec, repr=True)
    max_epochs: int = 1
    max_steps: Optional[int] = None
    patience: Optional[int] = 1
    improvement_delta: Optional[float] = 0.0
    holdout_ratio: float = 0.2
    max_holdout: Optional[int] = None

    def __post_init__(self):
        assert (
            not self.max_epochs or self.max_epochs > 0
        ), "Cannot train model for a negative number of epochs"
        assert not self.max_steps or self.max_steps > 0
        assert (
            self.max_epochs
            or self.max_steps
            or (self.improvement_delta is not None and self.patience)
        ), "Need at least one stopping criterion"
        assert (
            not self.patience or self.patience > 0
        ), "Must wait a positive number of epochs for any model to improve"
        assert (
            self.improvement_delta is None or self.improvement_delta >= 0
        ), "Improvement threshold must be nonnegative"
        assert self.holdout_ratio < 1.0, "Holdout data cannot be the entire dataset"
        assert (
            not self.max_holdout or self.max_holdout >= 0
        ), "Maximum number of holdout samples must be non-negative"

    def train_val_tensors(
        self, samples: SampleBatch, batch_keys: List[str], tensor_map_fn: callable
    ) -> Tuple[TensorDict, Optional[TensorDict]]:
        """Returns a tensor dict dataset split into training and validation.

        Shuffles the samples before splitting them.
        """
        total_count = samples.count
        holdout = int(total_count * self.holdout_ratio)
        if self.max_holdout is not None:
            holdout = min(holdout, self.max_holdout)

        samples.shuffle()
        train_data, eval_data = samples.slice(holdout, None), samples.slice(0, holdout)

        train_tensors = {k: tensor_map_fn(train_data[k]) for k in batch_keys}
        if eval_data.count == 0:
            eval_tensors = None
        else:
            eval_tensors = {k: tensor_map_fn(eval_data[k]) for k in batch_keys}

        return train_tensors, eval_tensors

    def train_val_loaders(
        self, train_tensors: TensorDict, val_tensors: Optional[TensorDict]
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train = self.dataloader.build_dataloader(train_tensors, train=True)
        val = (
            self.dataloader.build_dataloader(val_tensors, train=False)
            if val_tensors
            else None
        )
        return train, val


# ======================================================================================
# LightningModel
# ======================================================================================


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors,arguments-differ
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
        result = pl.TrainResult(loss)
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
# EarlyStopping
# ======================================================================================


class EarlyStopping(pl.callbacks.EarlyStopping):
    def __warn_deprecated_monitor_key(self):
        pass  # Disable annoying UserWarning


# ======================================================================================
# Policy Mixin
# ======================================================================================


class LightningModelMixin(ABC):
    """Adds model training behavior to a TorchPolicy class via PyTorch Lightning.

    Expects:
    * A `models` attribute in `self.module`
    * A `model_training` dict in `self.config`
    * A `model_warmup` dict in `self.config`
    * A 'models' optimizer in `self.optimizers`

    Attributes:
        model_training_spec: Specifications for training the model
        model_warmup_spec: Specifications for model warm-up
    """

    model_training_spec: TrainingSpec
    model_warmup_spec: TrainingSpec
    _pl_model: LightningModel = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_training_spec = TrainingSpec.from_dict(self.config["model_training"])
        self.model_warmup_spec = TrainingSpec.from_dict(self.config["model_warmup"])

    @property
    @abstractmethod
    def model_training_loss(self) -> Loss:
        """Loss function used for normal model training and evaluation.

        Must return a 1d Tensor with each model's losses and an info dict.
        """

    @property
    def model_warmup_loss(self) -> Loss:
        """Loss function used for model warm-up.

        Must return a 1d Tensor with each model's losses and an info dict.
        """
        return self.model_training_loss

    def optimize_model(
        self,
        samples: SampleBatch,
        warmup: bool = False,
    ) -> Tuple[List[float], StatDict]:
        """Update models with samples.

        Args:
            samples: Dataset as sample batches. Usually the entire replay buffer
            warmup: Whether to train with warm-up loss and spec

        Returns:
            A tuple with a list of each model's evaluation loss and a dictionary
            with training statistics
        """
        loss_fn = self.model_warmup_loss if warmup else self.model_training_loss
        spec = self.model_warmup_spec if warmup else self.model_training_spec

        train_tensors, val_tensors = spec.train_val_tensors(
            samples, loss_fn.batch_keys, self.convert_to_tensor
        )
        # Fit scalers for each model here
        for model in self.module.models:
            model.encoder.fit_scaler(
                train_tensors[SampleBatch.CUR_OBS], train_tensors[SampleBatch.ACTIONS]
            )

        model = self.get_lightning_model(loss_fn)
        trainer = self.get_trainer(spec)
        train, val = spec.train_val_loaders(train_tensors, val_tensors)

        info = self.run_training(model=model, trainer=trainer, train=train, val=val)
        info.update({"model_epochs": trainer.current_epoch + 1})
        return [np.nan for _ in self.module.models], info

    def get_lightning_model(self, loss_fn: Loss) -> LightningModel:
        if self._pl_model is None:
            self._pl_model = LightningModel(
                model=self.module.models,
                loss=loss_fn,
                optimizer=self.optimizers["models"],
            )

        self._pl_model.configure_losses(loss_fn)
        return self._pl_model

    @staticmethod
    def get_trainer(spec: TrainingSpec) -> pl.Trainer:
        return pl.Trainer(
            logger=False,
            early_stop_callback=EarlyStopping(
                min_delta=spec.improvement_delta,
                patience=spec.patience,
                mode="min",
                strict=False,
            ),
            max_epochs=spec.max_epochs,
            max_steps=spec.max_steps,
            progress_bar_refresh_rate=0,
            track_grad_norm=2,
            # gradient_clip_val=1e4,  # Broken
        )

    @staticmethod
    @supress_stderr
    @supress_stdout
    def run_training(model, trainer, train, val):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pytorch_lightning*")
            trainer.fit(model, train_dataloader=train, val_dataloaders=val)
            return trainer.test(model, test_dataloaders=val)[0]

    @staticmethod
    def model_training_defaults():
        """The default configuration dict for model training."""
        return TrainingSpec().to_dict()
