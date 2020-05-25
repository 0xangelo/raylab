"""Losses for Stochastic Value Gradients."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.utils.dictionaries import get_keys


class OneStepSVG:
    """Loss function for Stochastic Value Gradients with 1-step bootstrapping.

    Args:
        model (callable): stochastic model that reproduces state and its log density
        actor (callable): stochastic policy that reproduces action and its log density
        critic (callable): state-value function
        reward_fn (callable): reward function for state, action, and next state tuples
        gamma (float): discount factor
    """

    IS_RATIOS = "is_ratios"

    def __init__(self, model, actor, critic, reward_fn, **config):
        # pylint:disable=too-many-arguments
        self.model = model
        self.actor = actor
        self.critic = critic
        self.reward_fn = reward_fn
        self.config = config

    def __call__(self, batch):
        """Compute bootstrapped Stochatic Value Gradient loss."""
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

    def one_step_reproduced_state_value(self, obs, actions, next_obs, dones):
        """Compute 1-step approximation of the state value on real transition."""
        _acts, _ = self.actor(obs, actions)
        _next_obs, _ = self.model(obs, _acts, next_obs)
        _rewards = self.reward_fn(obs, _acts, _next_obs)
        _next_vals = self.critic(_next_obs).squeeze(-1)
        return torch.where(
            dones, _rewards, _rewards + self.config["gamma"] * _next_vals,
        )


class OneStepSoftSVG(OneStepSVG):
    """Loss function for bootstrapped maximum entropy Stochastic Value Gradients.

    Args:
        model (callable): stochastic model that reproduces state and its log density
        actor (callable): stochastic policy that reproduces action and its log density
        critic (callable): state-value function
        reward_fn (callable): reward function for state, action, and next state tuples
        alpha (callable): entropy coefficient schedule
        gamma (float): discount factor
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, *args, alpha, **config):
        # pylint:disable=too-many-arguments
        super().__init__(*args, **config)
        self.alpha = alpha

    @override(OneStepSVG)
    def one_step_reproduced_state_value(self, obs, actions, next_obs, dones):
        _actions, _logp = self.actor(obs, actions)
        _next_obs, _ = self.model(obs, _actions, next_obs)
        _rewards = self.reward_fn(obs, _actions, _next_obs)
        _entropy = -_logp
        _augmented_rewards = _rewards + _entropy * self.alpha()
        _next_vals = self.critic(_next_obs).squeeze(-1)

        gamma = self.config["gamma"]
        return torch.where(
            dones, _augmented_rewards, _augmented_rewards + gamma * _next_vals,
        )


class TrajectorySVG:
    """Loss function for Stochastic Value Gradients on full trajectory.

    Args:
        model (StochasticModel): model that reproduces state and its log density
        actor (StochasticActor): policy that reproduces action and its log density
        critic (callable): state-value function
        reward_fn (callable): reward function for state, action, and next state tuples
        gamma (float): discount factor
        torch_script (bool): whether to compile the rollout module to TorchScript
    """

    # pylint:disable=too-few-public-methods
    IS_RATIOS = "is_ratios"

    def __init__(self, model, actor, critic, reward_fn, **config):
        # pylint:disable=too-many-arguments
        self.critic = critic
        self.config = config
        rollout = ReproduceRollout(actor, model, reward_fn)
        self.rollout = torch.jit.script(rollout) if config["torch_script"] else rollout

    def __call__(self, episodes):
        """Compute Stochatic Value Gradient loss given full trajectories."""
        total_ret = 0
        for episode in episodes:
            init_obs = episode[SampleBatch.CUR_OBS][0]
            actions = episode[SampleBatch.ACTIONS]
            next_obs = episode[SampleBatch.NEXT_OBS]

            rewards, _ = self.rollout(actions, next_obs, init_obs)
            total_ret += rewards.sum()

        sim_return_mean = total_ret / len(episodes)
        loss = -sim_return_mean
        info = {"loss(actor)": loss.item(), "sim_return_mean": sim_return_mean.item()}
        return loss, info


class ReproduceRollout(nn.Module):
    """
    Neural network module that unrolls a policy, model and reward function
    given a trajectory.
    """

    def __init__(self, policy, model, reward_fn):
        super().__init__()
        self.policy = policy
        self.model = model
        self.reward_fn = reward_fn

    @override(nn.Module)
    def forward(self, acts, next_obs, init_ob):  # pylint: disable=arguments-differ
        reward_seq = []
        for act, next_ob in zip(acts, next_obs):
            _act, _ = self.policy.reproduce(init_ob, act)
            _next_ob, _ = self.model.reproduce(init_ob, _act, next_ob)
            reward_seq.append(self.reward_fn(init_ob, _act, _next_ob))
            init_ob = _next_ob
        return torch.stack(reward_seq), init_ob
