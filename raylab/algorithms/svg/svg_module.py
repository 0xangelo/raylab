"""Neural network modules for Stochastic Value Gradients."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class ParallelDynamicsModel(nn.Module):
    """
    Neural network module mapping inputs to distribution parameters through parallel
    subnetworks for each output dimension.
    """

    def __init__(self, *logits_modules):
        super().__init__()
        self.logits_modules = logits_modules
        self.loc_modules = [nn.Linear(m.out_features, 1) for m in self.logits_modules]
        self.scale_diag = nn.Parameter(torch.zeros(len(self.logits_modules)))

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = [m(obs, actions) for m in self.logits_modules]
        loc = torch.cat([m(i) for m, i in zip(self.loc_modules, logits)], dim=-1)
        return loc, self.scale_diag


class RecurrentPolicyModel(nn.Module):
    """
    Neural network module that unrolls a policy, model and reward function
    given a trajectory.
    """

    def __init__(
        self, policy_module, model_module, policy_dist, model_dist, reward_function
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.policy_module = policy_module
        self.model_module = model_module
        self.policy_dist = policy_dist
        self.model_dist = model_dist
        self.reward_function = reward_function

    @override(nn.Module)
    def forward(self, acts, next_obs, init_ob):
        # pylint: disable=arguments-differ,too-many-locals,consider-using-enumerate
        reward_seq = []
        for time in range(len(acts)):
            act, next_ob = acts[time], next_obs[time]
            pi_dist_params = self.policy_module(init_ob)
            pi_dist = self.policy_dist(*pi_dist_params)
            _act = pi_dist.reproduce(act)

            m_dist_params = self.model_module(init_ob, _act)
            m_dist = self.model_dist(*m_dist_params)
            _next_ob = m_dist.reproduce(next_ob)

            rew = self.reward_function(init_ob, _act)
            reward_seq.append(rew)

            init_ob = _next_ob
        return torch.stack(reward_seq), init_ob
