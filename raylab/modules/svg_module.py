"""SVG Architecture with disjoint model, actor, and critic."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


from raylab.distributions import DiagMultivariateNormal
from .basic import DistLogProb, DistReproduce, StateActionEncoder
from . import model_actor_critic as mac


class SVGModelActorCritic(mac.ModelActorCritic):
    """Module architecture with reparemeterized actor and model.

    Allows inference of noise variables given existing samples.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        if config.get("replay_kl") is False:
            old = self._make_actor(obs_space, action_space, config)
            self.old_actor = old["actor"].requires_grad_(False)

    @staticmethod
    @override(mac.ModelActorCritic)
    def _make_model_encoder(obs_space, action_space, config):
        if config["encoder"] == "svg_paper":
            return SVGDynamicsParams(obs_space, action_space, config)

        return mac.GaussianDynamicsParams(obs_space, action_space, config)

    @override(mac.ModelActorCritic)
    def _make_actor(self, obs_space, action_space, config):
        modules = super()._make_actor(obs_space, action_space, config)
        actor = modules["actor"]

        dist_logp = DistLogProb(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        dist_repr = DistReproduce(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        if config["torch_script"]:
            params_ = {
                "loc": torch.zeros(1, *action_space.shape),
                "scale_diag": torch.ones(1, *action_space.shape),
            }
            actions_ = torch.randn(1, *action_space.shape)
            dist_logp = dist_logp.traced(params_, actions_)
            dist_repr = dist_repr.traced(params_, actions_)

        actor.logp = PolicyLogProb(actor.params, dist_logp)
        actor.reproduce = PolicyReproduce(actor.params, dist_repr)
        return modules


class SVGDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to distribution parameters through parallel
    subnetworks for each output dimension.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        logits_modules = [
            StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                units=config["units"],
                delay_action=config["delay_action"],
                activation=config["activation"],
                **config["initializer_options"]
            )
            for _ in range(obs_space.shape[0])
        ]
        self.logits = nn.ModuleList(logits_modules)
        self.loc = nn.ModuleList([nn.Linear(m.out_features, 1) for m in self.logits])
        self.log_scale = nn.Parameter(torch.zeros(*obs_space.shape))

    @override(nn.Module)
    def forward(self, obs, act):  # pylint: disable=arguments-differ
        loc = torch.cat([l(m(obs, act)) for l, m in zip(self.loc, self.logits)], dim=-1)
        scale_diag = self.log_scale.exp().expand_as(loc)
        return {"loc": loc, "scale_diag": scale_diag}


class PolicyReproduce(nn.Module):
    """Reproduces observed actions."""

    def __init__(self, params_module, resample_module):
        super().__init__()
        self.params_module = params_module
        self.resample_module = resample_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.resample_module(self.params_module(obs), actions)


class PolicyLogProb(nn.Module):
    """Computes the log-likelihood of actions."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.logp_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.logp_module(self.params_module(obs), actions)
