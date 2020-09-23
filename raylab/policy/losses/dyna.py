# pylint:disable=missing-module-docstring
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.actor import StochasticPolicy
from raylab.policy.modules.critic import QValueEnsemble
from raylab.policy.modules.critic import VValue
from raylab.policy.modules.model import SME
from raylab.policy.modules.model import StochasticModel
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin
from .q_learning import QLearningMixin
from .utils import dist_params_stats


class StochasticPolicyAction(nn.Module):
    # pylint:disable=missing-docstring,arguments-differ,abstract-method
    def __init__(self, policy: StochasticPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: Tensor) -> Tensor:
        sample, _ = self.policy.sample(obs)
        return sample


class DeterministicPolicyAction(nn.Module):
    # pylint:disable=missing-docstring,arguments-differ,abstract-method
    def __init__(self, policy: DeterministicPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: Tensor) -> Tensor:
        return self.policy(obs)


class DynaQLearning(UniformModelPriorMixin, EnvFunctionsMixin, Loss):
    """Loss function Dyna-augmented Clipped Double Q-learning.

    Attributes:
        critics: Main action-values
        actor: Stochastic or deterministic policy
        models: Stochastic model ensemble
        target_critic: Target state-value function
        batch_keys: Keys required to be in the tensor batch
        gamma: discount factor
    """

    critics: QValueEnsemble
    actor: Union[DeterministicPolicy, StochasticPolicy]
    models: Union[StochasticModel, SME]
    target_critic: VValue

    gamma: float = 0.99
    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)
    _model_samples: int = 1

    def __init__(
        self,
        critics: QValueEnsemble,
        actor: Union[DeterministicPolicy, StochasticPolicy],
        models: Union[StochasticModel, SME],
        target_critic: VValue,
    ):
        super().__init__()
        self.critics = critics

        if isinstance(actor, DeterministicPolicy):
            self.get_action = DeterministicPolicyAction(actor)
        elif isinstance(actor, StochasticPolicy):
            self.get_action = StochasticPolicyAction(actor)
        else:
            raise ValueError(f"Unsupported actor type '{type(actor)}'")

        if isinstance(models, StochasticModel):
            # Treat everything as if ensemble
            models = SME([models])
        self.models = models
        self.target_critic = target_critic

        self._loss_fn = nn.MSELoss()

    @property
    def model_samples(self) -> int:
        """Number of next states to sample from model."""
        return self._model_samples

    @model_samples.setter
    def model_samples(self, value: int):
        assert value > 0, "Number of model samples must be positive"
        self._model_samples = value

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self._env.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]

        with torch.no_grad():
            action = self.get_action(obs)
            model, _ = self.sample_model()
            dist_params = model(obs, action)
            next_obs, _ = model.sample(dist_params, sample_shape=(self.model_samples,))

            reward = self._env.reward(obs, action, next_obs).mean(dim=0)
            next_values = self.target_critic(next_obs)
            dones = self._env.termination(obs, action, next_obs)
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)

            target = reward + self.gamma * next_values.mean(dim=0)

        values = self.critics(obs, action)
        loss = torch.stack([self._loss_fn(target, v) for v in values]).sum()

        stats = {"loss(critics)": loss.item()}
        stats.update(QLearningMixin.q_value_info(values))
        stats.update(dist_params_stats(dist_params, name="model"))
        return loss, stats
