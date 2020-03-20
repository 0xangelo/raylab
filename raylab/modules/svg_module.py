"""SVG Architecture with disjoint model, actor, and critic."""
import torch
import torch.nn as nn
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override

from .basic import StateActionEncoder
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin, GaussianDynamicsParams
from .stochastic_actor_mixin import StochasticActorMixin
from .state_value_mixin import StateValueMixin


BASE_CONFIG = {
    "torch_script": False,
    "mean_action_only": False,
    "actor": {
        "units": (32, 32),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    },
    "critic": {
        "units": (32, 32),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "target_vf": True,
    },
    "model": {
        "encoder": "svg_paper",
        "residual": True,
        "units": (32, 32),
        "activation": "Tanh",
        "delay_action": True,
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    },
}


class SVGModule(
    StochasticModelMixin,
    StochasticActorMixin,
    StateValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture with reparemeterized actor and model.

    Allows inference of noise variables given existing samples.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = merge_dicts(BASE_CONFIG, config)
        super().__init__(obs_space, action_space, config)
        if config.get("replay_kl") is False:
            old = self._make_actor(obs_space, action_space, config)
            self.old_actor = old["actor"].requires_grad_(False)

    @staticmethod
    @override(StochasticModelMixin)
    def _make_model_encoder(obs_space, action_space, config):
        if config["encoder"] == "svg_paper":
            return SVGDynamicsParams(obs_space, action_space, config)

        return GaussianDynamicsParams(obs_space, action_space, config)


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
