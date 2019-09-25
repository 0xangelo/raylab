"""Neural network modules for Stochastic Value Gradients."""
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal

__all__ = [
    "ParallelDynamicsModel",
    "DiagNormalLogProb",
    "DiagNormalEntropy",
    "DiagNormalRSample",
    "DiagNormalKL",
    "ReproduceRollout",
]


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


class DiagNormalLogProb(nn.Module):
    """
    Calculates the log-likelihood of an event given a Normal distribution's parameters.
    """

    @override(nn.Module)
    def forward(self, params, value):  # pylint: disable=arguments-differ
        return DiagMultivariateNormal(
            loc=params["loc"], scale_diag=params["scale_diag"]
        ).log_prob(value)


class DiagNormalEntropy(nn.Module):
    """
    Calculates the entropy given a Normal distribution's parameters.
    """

    @override(nn.Module)
    def forward(self, params):  # pylint: disable=arguments-differ
        return DiagMultivariateNormal(
            loc=params["loc"], scale_diag=params["scale_diag"]
        ).entropy()


class DiagNormalRSample(nn.Module):
    """Produce a reparametrized Normal sample, possibly with a desired value."""

    @override(nn.Module)
    def forward(self, params, value=None):  # pylint: disable=arguments-differ
        loc, scale_diag = params["loc"], params["scale_diag"]
        if value is None:
            eps = torch.randn_like(loc)
        else:
            with torch.no_grad():
                eps = (value - loc) / scale_diag
        return loc + scale_diag * eps


class DiagNormalKL(nn.Module):
    """
    Calculates the KL divergence between Normal distributions given their parameters.
    """

    @override(nn.Module)
    def forward(self, params, params_):  # pylint: disable=arguments-differ
        dist = DiagMultivariateNormal(
            loc=params["loc"], scale_diag=params["scale_diag"]
        )
        dist_ = DiagMultivariateNormal(
            loc=params_["loc"], scale_diag=params_["scale_diag"]
        )
        return kl_divergence(dist, dist_)


class ReproduceRollout(nn.Module):
    """
    Neural network module that unrolls a policy, model and reward function
    given a trajectory.
    """

    def __init__(
        self, policy_module, model_module, policy_rsample, model_rsample, reward_fn
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.policy_module = policy_module
        self.model_module = model_module
        self.policy_rsample = policy_rsample
        self.model_rsample = model_rsample
        self.reward_fn = reward_fn

    @override(nn.Module)
    def forward(self, acts, next_obs, init_ob):
        # pylint: disable=arguments-differ,too-many-locals,consider-using-enumerate
        reward_seq = []
        for act, next_ob in zip(acts, next_obs):
            pi_dist_params = self.policy_module(init_ob)
            _act = self.policy_rsample(pi_dist_params, act)

            m_dist_params = self.model_module(init_ob, _act)
            residual = self.model_rsample(m_dist_params, next_ob - init_ob)
            _next_ob = init_ob + residual

            rew = self.reward_fn(init_ob, _act, _next_ob)
            reward_seq.append(rew)

            init_ob = _next_ob
        return torch.stack(reward_seq), init_ob
