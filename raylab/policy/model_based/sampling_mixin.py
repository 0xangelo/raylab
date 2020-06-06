"""Environment model handling mixins for TorchPolicy."""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from numpy.random import Generator
from ray.rllib import SampleBatch
from torch.nn import Module


@dataclass(frozen=True)
class SamplingSpec(DataClassJsonMixin):
    """Specifications for sampling from the model.

    Attributes:
        num_elites: Use this number of best performing models to sample
            transitions
        rollout_length: Lenght of model-based rollouts from each initial
            state extracted from input sample batch
    """

    num_elites: int = 1
    rollout_length: int = 1

    def __post_init__(self):
        assert self.num_elites > 0, "Must have at least one elite model to sample from"
        assert (
            self.rollout_length > 0
        ), "Length of model-based rollouts must be positive"


class ModelSamplingMixin:
    """Adds model sampling behavior to a TorchPolicy class.

    Expects:
    * A `self.reward_fn` callable that computes the reward tensors for a batch
      of transitions
    * A `self.termination_fn` callable that computes the termination tensors for
      a batch of transitions
    * A `models` attribute in `self.module`
    * A `self.config` dict attribute
    * A `model_sampling` dict in `self.config`
    * A `seed` int in `self.config`

    Attributes:
        model_sampling_spec: Specifications for model training and sampling
        elite_models: Sequence of the `num_elites` best models sorted by
            performance. Initially set using the policy's model order.
        rng: Random number generator for choosing from the elite models for
            sampling.
    """

    model_sampling_spec: SamplingSpec
    elite_models: List[Module]
    rng: Generator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_sampling_spec = SamplingSpec.from_dict(self.config["model_sampling"])

        models = self.module.models
        num_elites = self.model_sampling_spec.num_elites
        assert num_elites <= len(models), "Cannot have more elites than models"

        self.elite_models = list(models[:num_elites])
        self.rng = np.random.default_rng(self.config["seed"])

    def setup_sampling_models(self, losses: List[float]):
        """Update the elite models based on model losses.

        Args:
            losses: list of model losses following the order of the ensemble
        """
        models = self.module.models
        self.elite_models = [models[i] for i in np.argsort(losses)]

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
        virtual_samples = []
        obs = init_obs = self.convert_to_tensor(samples[SampleBatch.CUR_OBS])

        for _ in range(self.model_sampling_spec.rollout_length):
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

    @staticmethod
    def model_sampling_defaults():
        """The default configuration dict for model sampling."""
        return SamplingSpec().to_dict()
