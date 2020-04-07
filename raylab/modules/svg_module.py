"""SVG Architecture with disjoint model, actor, and critic."""
import torch
import torch.nn as nn
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override

from .basic import NormalParams, StateActionEncoder
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin, GaussianDynamicsParams
from .stochastic_actor_mixin import StochasticActorMixin
from .state_value_mixin import StateValueMixin


BASE_CONFIG = {
    "torch_script": False,
    "actor": {
        "units": (32, 32),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
        "layer_norm": False,
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

        def make_encoder():
            return StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                units=config["units"],
                delay_action=config["delay_action"],
                activation=config["activation"],
                **config["initializer_options"]
            )

        self.logits = nn.ModuleList([make_encoder() for _ in range(obs_space.shape[0])])

        def make_param(in_features):
            kwargs = dict(event_dim=1, input_dependent_scale=False)
            return NormalParams(in_features, **kwargs)

        self.params = nn.ModuleList([make_param(l.out_features) for l in self.logits])

    @override(nn.Module)
    def forward(self, obs, act):  # pylint: disable=arguments-differ
        params = [p(l(obs, act)) for p, l in zip(self.params, self.logits)]
        loc = torch.cat([d["loc"] for d in params], dim=-1)
        scale = torch.cat([d["scale"] for d in params], dim=-1)
        return {"loc": loc, "scale": scale}
