"""Losses for computing policy gradients."""
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

from .utils import clipped_action_value


class DeterministicPolicyGradient:
    """Loss function for Deterministic Policy Gradient.

    Args:
        actor (callable): deterministic policy
        critics (list): callables for action-values
    """

    def __init__(self, actor, critics):
        self.actor = actor
        self.critics = critics

    def __call__(self, batch):
        obs = batch[SampleBatch.CUR_OBS]

        values = self.state_value(obs)
        loss = -torch.mean(values)

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    def state_value(self, obs):
        """Compute the state value by combining policy and action-value function."""
        actions = self.actor(obs)
        return clipped_action_value(obs, actions, self.critics)


class ReparameterizedSoftPG(DeterministicPolicyGradient):
    """Loss function for Soft Policy Iteration with reparameterized actor.

    Args:
        actor (StochasticPolicy): stochastic reparameterized policy
        critics (list): callables for action-values
        alpha (callable): entropy coefficient schedule
        rlogp (bool): whether to draw reparameterized log_probs from the actor
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, actor, critics, alpha):
        super().__init__(actor, critics)
        self.alpha = alpha

    @override(DeterministicPolicyGradient)
    def state_value(self, obs):
        actions, logp = self.actor.rsample(obs)
        action_values = clipped_action_value(obs, actions, self.critics)
        state_values = torch.mean(action_values - self.alpha() * logp)
        return state_values
