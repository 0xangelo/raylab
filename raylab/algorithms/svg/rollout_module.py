"""Neural network modules for Stochastic Value Gradients."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


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
