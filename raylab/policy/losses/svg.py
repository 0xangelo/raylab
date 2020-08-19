"""Losses for Stochastic Value Gradients."""
from typing import Callable
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import Tensor

from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.utils.annotations import RewardFn
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict
from raylab.utils.dictionaries import get_keys

from .abstract import Loss


StateRepr = Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]
ActionRepr = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


class OneStepSVG(Loss):
    """Loss function for Stochastic Value Gradients with 1-step bootstrapping.

    Args:
        model: stochastic model that reproduces state and its log density
        actor: stochastic policy that reproduces action and its log density
        critic: state-value function

    Attributes:
        gamma: discount factor
    """

    IS_RATIOS = "is_ratios"
    gamma: float = 0.99
    batch_keys: Tuple[str, str, str, str, str]

    def __init__(self, model: StateRepr, actor: ActionRepr, critic: nn.Module):
        self.model = model
        self.actor = actor
        self.critic = critic

        self.batch_keys = (
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
            self.IS_RATIOS,
        )
        self._reward_fn = None

    def set_reward_fn(self, function: RewardFn):
        """Set reward function to provided callable."""
        self._reward_fn = function

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute bootstrapped Stochatic Value Gradient loss."""
        assert (
            self._reward_fn is not None
        ), "No reward function set. Did you call `set_reward_fn`?"

        obs, actions, next_obs, dones, is_ratios = get_keys(
            batch,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
            self.IS_RATIOS,
        )
        state_val = self.one_step_reproduced_state_value(obs, actions, next_obs, dones)
        svg_loss = -torch.mean(is_ratios * state_val)
        return svg_loss, {"loss(actor)": svg_loss.item()}

    def one_step_reproduced_state_value(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor, dones: Tensor
    ) -> Tensor:
        """Compute 1-step approximation of the state value on real transition."""
        _acts, _ = self.actor(obs, actions)
        _next_obs, _ = self.model(obs, _acts, next_obs)
        _rewards = self._reward_fn(obs, _acts, _next_obs)
        _next_vals = self.critic(_next_obs).squeeze(-1)
        return torch.where(dones, _rewards, _rewards + self.gamma * _next_vals)


class OneStepSoftSVG(OneStepSVG):
    """Loss function for bootstrapped maximum entropy Stochastic Value Gradients.

    Args:
        model: stochastic model that reproduces state and its log density
        actor: stochastic policy that reproduces action and its log density
        critic: state-value function

    Attributes:
        gamma: discount factor
        alpha: entropy coefficient
    """

    gamma: float = 0.99
    alpha: float = 0.05

    @override(OneStepSVG)
    def one_step_reproduced_state_value(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor, dones: Tensor
    ) -> Tensor:
        _actions, _logp = self.actor(obs, actions)
        _next_obs, _ = self.model(obs, _actions, next_obs)
        _rewards = self._reward_fn(obs, _actions, _next_obs)
        _entropy = -_logp
        _augmented_rewards = _rewards + _entropy * self.alpha
        _next_vals = self.critic(_next_obs).squeeze(-1)

        return torch.where(
            dones, _augmented_rewards, _augmented_rewards + self.gamma * _next_vals,
        )


class TrajectorySVG(Loss):
    """Loss function for Stochastic Value Gradients on full trajectory.

    Args:
        model: model that reproduces state and its log density
        actor: policy that reproduces action and its log density
        critic: state-value function
    """

    batch_keys = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )

    def __init__(
        self, model: StochasticModel, actor: StochasticPolicy, critic: nn.Module
    ):
        self.model = model
        self.actor = actor
        self.critic = critic

        self._rollout = None

    def set_reward_fn(self, function: RewardFn):
        """Set reward function to provided callable."""
        self._rollout = ReproduceRewards(self.actor, self.model, function)

    def compile(self):
        """Compile the rollout module to TorchScript."""
        self._rollout = torch.jit.script(self._rollout)

    def __call__(self, episodes: List[TensorDict]) -> Tuple[Tensor, StatDict]:
        """Compute Stochatic Value Gradient loss given full trajectories."""
        assert (
            self._rollout is not None
        ), "Rollout module not set. Did you call `set_reward_fn`?"

        total_ret = 0
        for episode in episodes:
            init_obs = episode[SampleBatch.CUR_OBS][0]
            actions = episode[SampleBatch.ACTIONS]
            next_obs = episode[SampleBatch.NEXT_OBS]

            rewards = self._rollout(actions, next_obs, init_obs)
            total_ret += rewards.sum()

        sim_return_mean = total_ret / len(episodes)
        loss = -sim_return_mean
        info = {"loss(actor)": loss.item(), "sim_return_mean": sim_return_mean.item()}
        return loss, info


class ReproduceRewards(nn.Module):
    """Unrolls a policy, model and reward function given a trajectory.

    Args:
        model: transition model that reproduces state and its log density
        actor: stochastic policy that reproduces action and its log density
        reward_fn: reward function for state, action, and next state tuples
    """

    def __init__(
        self, policy: StochasticPolicy, model: StochasticModel, reward_fn: RewardFn
    ):
        super().__init__()
        self.policy = policy
        self.model = model
        self.reward_fn = reward_fn

    @override(nn.Module)
    def forward(self, acts: Tensor, next_obs: Tensor, init_ob: Tensor) -> Tensor:
        """Reproduce a sequence of actions, obsevations, and rewards.

        Args:
            acts: the sequence of actions
            next_obs: the sequence of observations that resulted from the
                application of the previous actions
            init_ob: the initial observation

        Returns:
            The rewards from the reproduced action sequence

        Note:
            Assumes the first tensor dimension of `acts` and `next_obs`
            corresponds to the timestep and iterates over it.
        """
        # pylint:disable=arguments-differ
        reward_seq = []
        for act, next_ob in zip(acts, next_obs):
            _act, _ = self.policy.reproduce(init_ob, act)
            _next_ob, _ = self.model.reproduce(next_ob, self.model(init_ob, _act))
            reward_seq.append(self.reward_fn(init_ob, _act, _next_ob))
            init_ob = _next_ob
        return torch.stack(reward_seq)
