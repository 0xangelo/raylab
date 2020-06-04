"""Environment model handling mixins for TorchPolicy."""
from __future__ import annotations

import collections
import copy
import itertools
import time
from dataclasses import dataclass
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from ray.rllib import SampleBatch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from raylab.pytorch.utils import TensorDictDataset


ModelSnapshot = collections.namedtuple("ModelSnapshot", "epoch loss state_dict")


@dataclass
class DataloaderSpec:
    """Specifications for creating the data loader."""

    batch_size: int = 256
    replacement: bool = False

    def __post_init__(self):
        assert self.batch_size > 0, "Model batch size must be positive"


@dataclass
class TrainingSpec:
    """Specifications for training the model."""

    dataloader: DataloaderSpec = DataloaderSpec()
    max_epochs: Optional[int] = 120
    max_grad_steps: Optional[int] = 120
    max_time: Optional[float] = 20
    patience_epochs: Optional[int] = 5
    improvement_threshold: float = 0.01

    @classmethod
    def from_dict(cls, dictionary: dict) -> TrainingSpec:
        """Instantiate a TrainingSpec from a dictionary."""
        dic = dictionary.copy()
        dic["dataloader"] = DataloaderSpec(**dic["dataloader"])
        return cls(**dic)

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


@dataclass
class ModelBasedSpec:
    """Specifications for model training and sampling."""

    training: TrainingSpec = TrainingSpec()
    num_elites: int = 5
    rollout_length: int = 1

    def __post_init__(self):
        assert self.num_elites > 0
        assert (
            self.rollout_length > 0
        ), "Length of model-based rollouts must be positive"

    @classmethod
    def from_dict(cls, dictionary: dict) -> ModelBasedSpec:
        """Instantiate a ModelBasedSpec from a dictionary."""
        dic = dictionary.copy()
        dic["training"] = TrainingSpec.from_dict(dic["training"])
        return cls(**dic)


class ModelBasedMixin:
    """Adds model related behavior to a TorchPolicy class.

    Expects a `models` attribute in `self.module` and a `model_based` config
    keyword.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        models = self.module.models
        num_elites = self.config["model_based"]["num_elites"]
        assert num_elites <= len(models), "Cannot have more elites than models"
        self.rng = np.random.default_rng(self.config["seed"])
        self.elite_models = self.rng.choice(models, size=num_elites, replace=False)

    def optimize_model(
        self, train_samples: SampleBatch, eval_samples: SampleBatch = None
    ) -> Dict[str, float]:
        """Update models with samples.

        Args:
            train_samples: training data
            eval_samples: holdout data

        Returns:
            A dictionary with training statistics
        """
        spec = ModelBasedSpec.from_dict(self.config["model_based"])
        snapshots = self._build_snapshots()
        dataloader = self._build_dataloader(train_samples, spec.training.dataloader)
        eval_tensors = self._lazy_tensor_dict(eval_samples) if eval_samples else None

        info, snapshots = self._train_model_epochs(
            dataloader, snapshots, eval_tensors, spec.training
        )
        info.update(self._restore_models_and_set_elites(snapshots))

        info.update(self.extra_grad_info("models"))
        return self._learner_stats(info)

    def _build_snapshots(self) -> List[ModelSnapshot]:
        return [
            ModelSnapshot(epoch=0, loss=None, state_dict=copy.deepcopy(m.state_dict()))
            for m in self.module.models
        ]

    def _build_dataloader(self, train_samples: SampleBatch, spec: DataloaderSpec):
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
                    loss, _ = self.loss_model(minibatch)
                    loss.backward()
                grad_steps += 1
                if spec.max_grad_steps and grad_steps >= spec.max_grad_steps:
                    break

            if eval_tensors:
                with torch.no_grad():
                    _, eval_info = self.loss_model(eval_tensors)

                snapshots = self._update_snapshots(epoch, snapshots, eval_info, spec)
                info.update(eval_info)

            if self._terminate_epoch(epoch, snapshots, start, grad_steps, spec):
                break

        info["model_epochs"] = epoch + 1
        return info, snapshots

    @staticmethod
    def _model_epochs(spec: TrainingSpec) -> Iterator[int]:
        return iter(range(spec.max_epochs)) if spec.max_epochs else itertools.count()

    def _update_snapshots(
        self,
        epoch: int,
        snapshots: List[ModelSnapshot],
        info: Dict[str, float],
        spec: TrainingSpec,
    ) -> List[ModelSnapshot]:
        def update_snapshot(idx, snap):
            cur_loss = info[f"loss(models[{idx}])"]
            threshold = spec.improvement_threshold
            if snap.loss is None or (snap.loss - cur_loss) / snap.loss > threshold:
                return ModelSnapshot(
                    epoch=epoch,
                    loss=cur_loss,
                    state_dict=copy.deepcopy(self.module.models[idx].state_dict()),
                )
            return snap

        return [update_snapshot(i, s) for i, s in enumerate(snapshots)]

    @staticmethod
    def _terminate_epoch(
        epoch: int,
        snapshots: List[ModelSnapshot],
        start_time_s: float,
        model_steps: int,
        spec: TrainingSpec,
    ) -> bool:
        patience_epochs = spec.patience_epochs or float("inf")
        max_train_s = spec.max_train_s or float("inf")
        max_grad_steps = spec.max_grad_steps or float("inf")

        return (
            time.time() - start_time_s >= max_train_s
            or epoch - max(s.epoch for s in snapshots) >= patience_epochs
            or model_steps >= max_grad_steps
        )

    def _restore_models_and_set_elites(
        self, snapshots: List[ModelSnapshot]
    ) -> Dict[str, float]:
        info = {}
        for idx, snap in enumerate(snapshots):
            self.module.models[idx].load_state_dict(snap.state_dict)
            info[f"loss(models[{idx}])"] = snap.loss

        num_elites = len(self.elite_models)
        elite_idxs = np.argsort([s.loss for s in snapshots])[:num_elites]
        info["loss(models[elites])"] = np.mean([snapshots[i].loss for i in elite_idxs])
        self.elite_models = [self.module.models[i] for i in elite_idxs]
        return info

    @torch.no_grad()
    def generate_virtual_sample_batch(self, samples: SampleBatch) -> SampleBatch:
        """Rollout model with latest policy.

        Produces samples for populating the virtual buffer, hence no gradient
        information is retained.

        If a transition is terminal, the next transition, if any, is generated from
        the initial state passed through `samples`.

        Args:
            samples: the transitions to extract initial states from

        Returns:
            A batch of transitions sampled from the model
        """
        spec = ModelBasedSpec.from_dict(self.config["model_based"])
        virtual_samples = []
        obs = init_obs = self.convert_to_tensor(samples[SampleBatch.CUR_OBS])

        for _ in range(spec.rollout_length):
            model = self.rng.choice(self.elite_models)

            action, _ = self.module.actor.sample(obs)
            next_obs, _ = model.sample(obs, action)
            reward = self.reward_fn(obs, action, next_obs)
            done = self.termination_fn(obs, action, next_obs)

            transition = {
                SampleBatch.CUR_OBS: obs,
                SampleBatch.ACTIONS: action,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.REWARDS: reward,
                SampleBatch.DONES: done,
            }
            virtual_samples += [
                SampleBatch({k: v.numpy() for k, v in transition.items()})
            ]
            obs = torch.where(done.unsqueeze(-1), init_obs, next_obs)

        return SampleBatch.concat_samples(virtual_samples)
