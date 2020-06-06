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
from dataclasses_json import DataClassJsonMixin
from ray.rllib import SampleBatch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from raylab.pytorch.utils import TensorDictDataset


ModelSnapshot = collections.namedtuple("ModelSnapshot", "epoch loss state_dict")


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
    improvement_threshold: float = 0.01

    def __post_init__(self):
        assert (
            not self.max_epochs or self.max_epochs > 0
        ), "Cannot train model for a negative number of epochs"
        assert not self.max_grad_steps or self.max_grad_steps > 0
        assert (
            not self.patience_epochs or self.patience_epochs > 0
        ), "Must wait a positive number of epochs for any model to improve"
        assert (
            not self.max_time or self.max_time > 0
        ), "Maximum training time must be positive"

        assert (
            self.max_epochs or self.max_grad_steps or self.patience_epochs
        ), "Need at least one stopping criterion"


class ModelTrainingMixin:
    """Adds model related behavior to a TorchPolicy class.

    Expects:
    * A `models` attribute in `self.module`
    * A `config` dict attribute
    * A `model_training` dict in `self.config`
    * An `optimizer` attribute with subattribute `models`
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

        Args:
            train_samples: training data
            eval_samples: holdout data

        Returns:
            A tuple with a list of each model's evaluation loss and a dictionary
            with training statistics
        """
        spec = self.model_training_spec
        snapshots = self._build_snapshots()
        dataloader = self._build_dataloader(train_samples, spec.dataloader)
        eval_tensors = self._lazy_tensor_dict(eval_samples) if eval_samples else None

        info, snapshots = self._train_model_epochs(
            dataloader, snapshots, eval_tensors, spec
        )

        info.update(self._restore_models(snapshots))
        info.update(self.model_grad_info())
        eval_losses = [s.loss if s.loss else np.nan for s in snapshots]
        return eval_losses, info

    def _build_snapshots(self) -> List[ModelSnapshot]:
        return [
            ModelSnapshot(epoch=0, loss=None, state_dict=copy.deepcopy(m.state_dict()))
            for m in self.module.models
        ]

    def _build_dataloader(
        self, train_samples: SampleBatch, spec: DataloaderSpec
    ) -> DataLoader:
        train_tensors = self._lazy_tensor_dict(train_samples)
        dataset = TensorDictDataset(
            {k: train_tensors[k] for k in self.loss_model.batch_keys}
        )
        sampler = RandomSampler(dataset, replacement=spec.replacement)
        return DataLoader(dataset, sampler=sampler, batch_size=spec.batch_size)

    def _train_model_epochs(
        self,
        dataloader: DataLoader,
        snapshots: List[ModelSnapshot],
        eval_tensors: Dict[str, torch.Tensor],
        spec: TrainingSpec,
    ) -> Tuple[Dict[str, float], List[ModelSnapshot]]:

        info = {}
        grad_steps = 0
        start = time.time()
        epoch = -1
        for epoch in self._model_epochs(spec):
            for minibatch in dataloader:
                with self.optimizer.models.optimize():
                    losses, train_info = self.loss_model(minibatch)
                    losses.mean().backward()

                info.update({"train_" + k: v for k, v in train_info.items()})
                grad_steps += 1
                if spec.max_grad_steps and grad_steps >= spec.max_grad_steps:
                    break

            if eval_tensors:
                with torch.no_grad():
                    eval_losses, eval_info = self.loss_model(eval_tensors)
                    eval_losses = eval_losses.tolist()

                snapshots = self._updated_snapshots(snapshots, epoch, eval_losses, spec)
                info.update({"eval_" + k: v for k, v in eval_info.items()})

            if self._terminate_epoch(epoch, snapshots, start, grad_steps, spec):
                break

        info["model_epochs"] = epoch + 1
        return info, snapshots

    @staticmethod
    def _model_epochs(spec: TrainingSpec) -> Iterator[int]:
        return iter(range(spec.max_epochs)) if spec.max_epochs else itertools.count()

    def _updated_snapshots(
        self,
        snapshots: List[ModelSnapshot],
        epoch: int,
        losses: List[float],
        spec: TrainingSpec,
    ) -> List[ModelSnapshot]:
        threshold = spec.improvement_threshold

        def updated_snapshot(model, snap, cur_loss):
            if snap.loss is None or (snap.loss - cur_loss) / snap.loss > threshold:
                return ModelSnapshot(
                    epoch=epoch,
                    loss=cur_loss,
                    state_dict=copy.deepcopy(model.state_dict()),
                )
            return snap

        return [
            updated_snapshot(model=m, snap=s, cur_loss=l)
            for m, s, l in zip(self.module.models, snapshots, losses)
        ]

    @staticmethod
    def _terminate_epoch(
        epoch: int,
        snapshots: List[ModelSnapshot],
        start_time: float,
        model_steps: int,
        spec: TrainingSpec,
    ) -> bool:
        patience_epochs = spec.patience_epochs or float("inf")
        max_time = spec.max_time or float("inf")
        max_grad_steps = spec.max_grad_steps or float("inf")

        return (
            time.time() - start_time >= max_time
            or epoch - max(s.epoch for s in snapshots) >= patience_epochs
            or model_steps >= max_grad_steps
        )

    def _restore_models(self, snapshots: List[ModelSnapshot]) -> Dict[str, float]:
        info = {}
        for idx, snap in enumerate(snapshots):
            self.module.models[idx].load_state_dict(snap.state_dict)
            info[f"loss(models[{idx}])"] = snap.loss
        return info

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
