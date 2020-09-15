# pylint:disable=missing-docstring
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
from ray.rllib import SampleBatch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from raylab.options import option
from raylab.policy.losses import Loss
from raylab.policy.modules.model import SME
from raylab.torch.utils import convert_to_tensor
from raylab.torch.utils import TensorDictDataset
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict
from raylab.utils.lightning import supress_stderr
from raylab.utils.lightning import supress_stdout


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
    _epoch_outputs: List[Tuple[Tensor, StatDict]]
    _best_outputs: Tuple[List[float], StatDict]
    _module_state: dict

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self._best_outputs = None
        self.save_module_state(pl_module)

    def __warn_deprecated_monitor_key(self):
        pass  # Disable annoying UserWarning

    def on_validation_epoch_start(self, trainer, pl_module):
        self._epoch_outputs = []
        super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        # pylint:disable=too-many-arguments
        self._epoch_outputs += [pl_module.val_loss.last_output]
        super().on_validation_batch_end(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )

    def on_validation_end(self, trainer, pl_module):
        is_start = not torch.isfinite(self.best_score).item()
        super().on_validation_end(trainer, pl_module)
        if self.wait_count == 0 and not is_start:  # Improved
            self.save_outputs()
            self.save_module_state(pl_module)
        elif self._best_outputs is None:
            self.save_outputs()

    def save_outputs(self):
        epoch_losses, epoch_infos = zip(*self._epoch_outputs)
        model_losses = torch.stack(epoch_losses, dim=0).mean(dim=0).tolist()
        model_infos = {k: stats.mean(i[k] for i in epoch_infos) for k in epoch_infos[0]}
        self._best_outputs = (model_losses, model_infos)

    def save_module_state(self, pl_module):
        self._module_state = copy.deepcopy(pl_module.state_dict())

    def state_dict(self):
        state = super().state_dict()
        state.update(best_outputs=self._best_outputs, module=self._module_state)
        return state

    def load_state_dict(self, state_dict):
        self._best_outputs = state_dict["best_outputs"]
        self._module_state = copy.deepcopy(state_dict["module"])
        used = set("best_outputs module".split())
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in used})


# ======================================================================================
# Policy Mixin
# ======================================================================================


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

    def build_trainer(self, check_val: bool) -> pl.Trainer:
        return pl.Trainer(
            logger=False,
            num_sanity_val_steps=2 if check_val else 0,
            early_stop_callback=EarlyStopping(
                min_delta=self.improvement_delta,
                patience=self.patience,
                mode="min",
                strict=False,
            ),
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            progress_bar_refresh_rate=0,
            track_grad_norm=2,
            # gradient_clip_val=1e4,  # Broken
        )


class LightningModelTrainer:
    """Model training behavior for TorchPolicy instances via PyTorch Lightning.

    Args:
        models: Stochastic model ensemble
        loss_fn: Loss associated with the model ensemble
        optimizer: Optimizer associated with the model ensemble
        config: Dictionary containg `model_training` and `model_warmup` dicts

    Attributes:
        model_training_spec: Specifications for training the model
        model_warmup_spec: Specifications for model warm-up
        model_training_loss: Loss function used for normal model training and
            evaluation
        model_warmup_loss: Loss function used for model warm-up.
    """

    model_training_spec: TrainingSpec
    model_warmup_spec: TrainingSpec
    pl_model: LightningModel

    def __init__(self, models: SME, loss_fn: Loss, optimizer: Optimizer, config: dict):
        self.model_training_spec = TrainingSpec.from_dict(config["model_training"])
        self.model_warmup_spec = TrainingSpec.from_dict(config["model_warmup"])
        self.model_training_loss = self.model_warmup_loss = loss_fn
        self.pl_model = LightningModel(model=models, loss=loss_fn, optimizer=optimizer)

    def optimize(
        self, samples: SampleBatch, warmup: bool = False
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

        pl_model = self.pl_model
        pl_model.configure_losses(loss_fn)
        train_tensors, val_tensors = spec.train_val_tensors(
            samples,
            loss_fn.batch_keys,
            lambda x: convert_to_tensor(x, device=pl_model.device),
        )
        # Fit scalers for each model here
        for model in pl_model.model:
            model.encoder.fit_scaler(
                train_tensors[SampleBatch.CUR_OBS], train_tensors[SampleBatch.ACTIONS]
            )

        trainer = spec.build_trainer(check_val=warmup)
        train, val = spec.train_val_loaders(train_tensors, val_tensors)

        losses, info = self.run_training(
            model=self.pl_model, trainer=trainer, train=train, val=val
        )
        return losses, info

    @staticmethod
    @supress_stderr
    @supress_stdout
    def run_training(
        model: LightningModel, trainer: pl.Trainer, train: DataLoader, val: DataLoader
    ) -> Tuple[List[float], StatDict]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pytorch_lightning*")
            trainer.fit(model, train_dataloader=train, val_dataloaders=val)

        saved_state = trainer.early_stop_callback.state_dict()
        model.load_state_dict(saved_state["module"])
        losses, info = saved_state["best_outputs"]
        info.update({"model_epochs": trainer.current_epoch + 1})
        return losses, info

    @staticmethod
    def add_options(cls_: type) -> type:
        """Add options for classes that may use this class."""
        cls = cls_
        for opt in [
            option(
                "model_training",
                default=TrainingSpec().to_dict(),
                help=TrainingSpec.__doc__,
            ),
            option(
                "model_warmup",
                default=TrainingSpec().to_dict(),
                help="Specifications for model warm-up; same as 'model_training'",
            ),
        ]:
            cls = opt(cls)
        return cls
