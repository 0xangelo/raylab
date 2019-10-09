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
        self.logits_modules = nn.ModuleList(logits_modules)
        self.loc_modules = nn.ModuleList(
            [nn.Linear(m.out_features, 1) for m in self.logits_modules]
        )
        self.log_scale = nn.Parameter(torch.zeros(len(self.logits_modules)))

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = [m(obs, actions) for m in self.logits_modules]
        loc = torch.cat([m(i) for m, i in zip(self.loc_modules, logits)], dim=-1)
        scale_diag = self.log_scale.exp()
        return dict(loc=loc, scale_diag=scale_diag)


class ModelReproduce(nn.Module):
    """Reproduces observed transitions."""

    def __init__(self, params_module, resample_module):
        super().__init__()
        self.params_module = params_module
        self.resample_module = resample_module

    @override(nn.Module)
    def forward(self, obs, actions, new_obs):  # pylint: disable=arguments-differ
        residual = self.resample_module(self.params_module(obs, actions), new_obs - obs)
        return obs + residual


class ModelLogProb(nn.Module):
    """Computes the log-likelihood of transitions."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.logp_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions, new_obs):  # pylint: disable=arguments-differ
        residual = new_obs - obs
        return self.logp_module(self.params_module(obs, actions), residual)


class PolicyReproduce(nn.Module):
    """Reproduces observed actions."""

    def __init__(self, params_module, resample_module):
        super().__init__()
        self.params_module = params_module
        self.resample_module = resample_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        return self.resample_module(self.params_module(obs), actions)


class PolicyLogProb(nn.Module):
    """Computes the log-likelihood of actions."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.logp_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        return self.logp_module(self.params_module(obs), actions)


class ReproduceRollout(nn.Module):
    """
    Neural network module that unrolls a policy, model and reward function
    given a trajectory.
    """

    def __init__(self, policy_reproduce, model_reproduce, reward_fn):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.policy_reproduce = policy_reproduce
        self.model_reproduce = model_reproduce
        self.reward_fn = reward_fn

    @override(nn.Module)
    def forward(self, acts, next_obs, init_ob):
        # pylint: disable=arguments-differ,too-many-locals,consider-using-enumerate
        reward_seq = []
        for act, next_ob in zip(acts, next_obs):
            _act = self.policy_reproduce(init_ob, act)
            _next_ob = self.model_reproduce(init_ob, _act, next_ob)
            reward_seq.append(self.reward_fn(init_ob, _act, _next_ob))
            init_ob = _next_ob
        return torch.stack(reward_seq), init_ob
