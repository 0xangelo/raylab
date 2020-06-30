"""Environment model handling mixins for TorchPolicy."""
import collections
import copy
import itertools
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from ray.rllib import SampleBatch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from raylab.policy.losses.abstract import Loss
from raylab.pytorch.utils import TensorDictDataset


@dataclass(frozen=True)
class DataloaderSpec(DataClassJsonMixin):
    """Specifications for creating the data loader.

    Attributes:
        batch_size: Size of minibatch for dynamics model training
        replacement: Whether to sample transitions with replacement
    """

    batch_size: int = 256
    replacement: bool = False

    def __post_init__(self):
        assert self.batch_size > 0, "Model batch size must be positive"


@dataclass(frozen=True)
class TrainingSpec(DataClassJsonMixin):
    """Specifications for training the model.

    Attributes:
        dataloader: specifications for creating the data loader
        max_epochs: Maximum number of full model passes through the data
        max_grad_steps: Maximum number of model gradient steps
        improvement_threshold: Minimum expected relative improvement in model
            validation loss
        patience_epochs: Number of epochs to wait for any of the models to
            improve on the validation dataset before early stopping
        max_time: Maximum time in seconds for training the model. We
            check this after each epoch (not minibatch)
    """

    dataloader: DataloaderSpec = field(default_factory=DataloaderSpec, repr=True)
    max_epochs: Optional[int] = 120
    max_grad_steps: Optional[int] = 120
    max_time: Optional[float] = 20
    patience_epochs: Optional[int] = 5
    improvement_threshold: Optional[float] = 0.01

    def __post_init__(self):
        assert (
            not self.max_epochs or self.max_epochs > 0
        ), "Cannot train model for a negative number of epochs"
        assert not self.max_grad_steps or self.max_grad_steps > 0
        assert (
            not self.max_time or self.max_time > 0
        ), "Maximum training time must be positive"
        assert (
            not self.patience_epochs or self.patience_epochs > 0
        ), "Must wait a positive number of epochs for any model to improve"
        assert (
            self.improvement_threshold is None or self.improvement_threshold >= 0
        ), "Improvement threshold must be nonnegative"
        assert (
            self.max_epochs
            or self.max_grad_steps
            or (self.improvement_threshold is not None and self.patience_epochs)
        ), "Need at least one stopping criterion"


ModelSnapshot = collections.namedtuple("ModelSnapshot", "epoch loss state_dict")


@dataclass
class Evaluator:
    """Evaluates models and saves snapshots.

    Holds snapshots for each model. A snapshot contains the epoch in which the
    model was evaluated, its validation loss, and its state dict.

    Note:
        Upon creation, evaluates models on validation data to set initial
        snapshots.

    Args:
        models: the model ensemble
        loss_fn: the loss function for model ensemble
        improvement_threshold: Minimum expected relative improvement in model
            validation loss
        patience_epochs: Number of epochs to wait for any of the models to
            improve on the validation dataset before early stopping. If None,
            disables eary stopping.
    """

    models: nn.ModuleList
    loss_fn: Loss
    eval_tensors: Dict[str, Tensor]
    improvement_threshold: float
    patience_epochs: Optional[int]

    def __post_init__(self):
        eval_losses, _ = self.loss_fn(self.eval_tensors)
        eval_losses = eval_losses.tolist()
        self._snapshots = [
            ModelSnapshot(epoch=-1, loss=loss, state_dict=copy.deepcopy(m.state_dict()))
            for m, loss in zip(self.models, eval_losses)
        ]

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[bool, Dict[str, float]]:
        """Evaluate models on holdout data and update snapshots.

        Args:
            epoch: the epoch number

        Returns:
            A tuple with two values: whether or not to early stop training based
            on validation loss improvement and a dict with validation loss info
        """
        eval_losses, eval_info = self.loss_fn(self.eval_tensors)
        eval_losses = eval_losses.tolist()
        eval_info = {"eval_" + k: v for k, v in eval_info.items()}

        self._update_snapshots(epoch, eval_losses)

        patience_epochs = self.patience_epochs or float("inf")
        early_stop = epoch - max(s.epoch for s in self._snapshots) >= patience_epochs
        return early_stop, eval_info

    def _update_snapshots(self, epoch: int, eval_losses: List[float]):
        snapshots = self._snapshots
        threshold = self.improvement_threshold

        def updated_snapshot(model, snap, cur_loss):
            if (snap.loss - cur_loss) / snap.loss > threshold:
                return ModelSnapshot(
                    epoch=epoch,
                    loss=cur_loss,
                    state_dict=copy.deepcopy(model.state_dict()),
                )
            return snap

        self._snapshots = [
            updated_snapshot(model=m, snap=s, cur_loss=l)
            for m, s, l in zip(self.models, snapshots, eval_losses)
        ]

    def restore_models(self) -> List[float]:
        """Restore models to the best performing parameters.

        Returns:
            A list with the validation performances of each model
        """
        losses = []
        for idx, snap in enumerate(self._snapshots):
            self.models[idx].load_state_dict(snap.state_dict)
            losses += [snap.loss]
        return losses


class ModelTrainingMixin:
    """Adds model training behavior to a TorchPolicy class.

    Expects:
    * A `models` attribute in `self.module`
    * A `model_training` dict in `self.config`
    * A 'models' optimizer in `self.optimizers`
    * A `loss_model` callable attribute that returns a 1d Tensor with each
      model's losses and an info dict

    Attributes:
        model_training_spec: Specifications for training the model
    """

    model_training_spec: TrainingSpec

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_training_spec = TrainingSpec.from_dict(self.config["model_training"])

    def optimize_model(
        self, train_samples: SampleBatch, eval_samples: SampleBatch = None
    ) -> Tuple[List[float], Dict[str, float]]:
        """Update models with samples.

        If `spec.max_epochs` is set, training will be cut off after this many
        epochs.

        If `spec.max_grad_steps` is set, training will be cut off after this
        many model gradient steps.

        If `spec.max_time` is set, it will cut off training after this many
        seconds have elapsed.

        If `spec.improvement_threshold` is set, it will save snapshots based on
        model performance on validation data and restore models to the
        best performing parameters after training. Otherwise, it will return the
        latest models.

        If `spec.patience_epochs` is set, it will wait for at most this many
        epochs for any model to improve on the validation dataset, otherwise it
        will stop training.

        Args:
            train_samples: training data
            eval_samples: holdout data

        Returns:
            A tuple with a list of each model's evaluation loss and a dictionary
            with training statistics
        """
        dataloader = self._build_dataloader(train_samples)
        evaluator = self._setup_evaluator(eval_samples)

        info = self._train_model_epochs(dataloader, evaluator)

        if evaluator:
            eval_losses = evaluator.restore_models()
            info.update(
                {f"eval_loss(models[{i}])": l for i, l in enumerate(eval_losses)}
            )
        else:
            eval_losses = [np.nan for _ in self.module.models]

        info.update(self.model_grad_info())
        return eval_losses, info

    def _build_dataloader(self, train_samples: SampleBatch) -> DataLoader:
        spec = self.model_training_spec.dataloader
        train_tensors = {
            k: self.convert_to_tensor(train_samples[k])
            for k in self.loss_model.batch_keys
        }
        dataset = TensorDictDataset(train_tensors)
        sampler = RandomSampler(dataset, replacement=spec.replacement)
        return DataLoader(dataset, sampler=sampler, batch_size=spec.batch_size)

    def _setup_evaluator(
        self, eval_samples: Optional[SampleBatch]
    ) -> Optional[Evaluator]:
        spec = self.model_training_spec
        if not (eval_samples and spec.improvement_threshold is not None):
            return None

        eval_tensors = {
            k: self.convert_to_tensor(eval_samples[k])
            for k in self.loss_model.batch_keys
        }
        return Evaluator(
            self.module.models,
            self.loss_model,
            eval_tensors,
            spec.improvement_threshold,
            spec.patience_epochs,
        )

    def _train_model_epochs(
        self, dataloader: DataLoader, evaluator: Optional[Evaluator],
    ) -> Dict[str, float]:

        spec = self.model_training_spec
        info = {}
        grad_steps = 0
        start = time.time()
        early_stop = False
        epoch = -1
        for epoch in self._model_epochs(spec):
            for minibatch in dataloader:
                with self.optimizers.optimize("models"):
                    losses, train_info = self.loss_model(minibatch)
                    losses.sum().backward()

                info.update({"train_" + k: v for k, v in train_info.items()})
                grad_steps += 1
                if spec.max_grad_steps and grad_steps >= spec.max_grad_steps:
                    break

            if evaluator:
                early_stop, eval_info = evaluator.validate(epoch)
                info.update(eval_info)

            if early_stop or self._terminate_epoch(start, grad_steps, spec):
                break

        info["model_epochs"] = epoch + 1
        return info

    @staticmethod
    def _model_epochs(spec: TrainingSpec) -> Iterator[int]:
        return iter(range(spec.max_epochs)) if spec.max_epochs else itertools.count()

    @staticmethod
    def _terminate_epoch(
        start_time: float, model_steps: int, spec: TrainingSpec,
    ) -> bool:
        max_time = spec.max_time or float("inf")
        max_grad_steps = spec.max_grad_steps or float("inf")

        return time.time() - start_time >= max_time or model_steps >= max_grad_steps

    @torch.no_grad()
    def model_grad_info(self) -> Dict[str, float]:
        """Returns the average gradient norm accross models."""
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()
            for m in self.module.models
        ]
        return {"grad_norm(models)": np.mean(grad_norms)}

    @staticmethod
    def model_training_defaults():
        """The default configuration dict for model training."""
        return TrainingSpec().to_dict()
