"""Loss functions for Decision-Aware Model Learning."""
import torch
from ray.rllib import SampleBatch

from .utils import clipped_action_value


class DPGAwareModelLearning:
    """Loss function for Deterministic Policy Gradient-Aware model learning.

    Args:
        model (callable): stochastic model that returns next state and its log density
        actor (callable): deterministic policy
        critics (list): callables for action-values
        reward_fn (callable): reward function for state, action, and next state tuples
        gamma (float): discount factor
        grad_estimator (str): one of 'PD' or 'SF'
    """

    # pylint:disable=too-many-arguments
    def __init__(self, model, actor, critics, reward_fn, **config):
        self.model = model
        self.actor = actor
        self.critics = critics
        self.reward_fn = reward_fn
        self.config = config

    def __call__(self, batch):
        """Compute policy gradient-aware (PGA) model loss."""
        obs = batch[SampleBatch.CUR_OBS]
        actions = self.actor(obs).detach().requires_grad_()

        predictions = self.one_step_action_value_surrogate(obs, actions)
        targets = self.zero_step_action_values(obs, actions)

        temporal_diff = torch.sum(targets - predictions)
        (action_gradients,) = torch.autograd.grad(
            temporal_diff, actions, create_graph=True
        )

        daml_loss = torch.sum(action_gradients * action_gradients, dim=-1).mean()
        return (
            daml_loss,
            {"loss(action)": temporal_diff.item(), "loss(model)": daml_loss.item()},
        )

    def one_step_action_value_surrogate(self, obs, actions, model_samples=1):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Deterministic Policy Gradient
        using target networks and model transitions.
        """
        next_obs, rewards, logp = self._generate_transition(obs, actions, model_samples)
        # Next action grads shouldn't propagate
        with torch.no_grad():
            next_acts = self.actor(next_obs)
        next_values = clipped_action_value(next_obs, next_acts, self.critics)
        values = rewards + self.config["gamma"] * next_values

        if self.config["grad_estimator"] == "SF":
            surrogate = torch.mean(logp * values.detach(), dim=0)
        elif self.config["grad_estimator"] == "PD":
            surrogate = torch.mean(values, dim=0)
        return surrogate

    def _generate_transition(self, obs, actions, num_samples):
        """Compute virtual transition and its log density."""
        sample_shape = (num_samples,)
        obs = obs.expand(sample_shape + obs.shape)
        actions = actions.expand(sample_shape + actions.shape)

        next_obs, logp = self.model(obs, actions)
        rewards = self.reward_fn(obs, actions, next_obs)
        return next_obs, rewards, logp

    def zero_step_action_values(self, obs, actions):
        """Compute Q^{\\pi}(s, a) directly using approximate critic."""
        return clipped_action_value(obs, actions, self.critics)
