"""SVG Architecture with disjoint model, actor, and critic."""
import torch
import torch.nn as nn
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override

from .basic import NormalParams, StateActionEncoder
from .distributions import Independent, Normal
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModel
from .stochastic_actor_mixin import StochasticActorMixin
from .state_value_mixin import StateValueMixin


BASE_MODEL_CONFIG = {
    "residual": True,
    "input_dependent_scale": False,
    "encoder": {
        "units": (40, 40),
        "activation": "Tanh",
        "delay_action": True,
        "initializer_options": {"name": "xavier_uniform"},
    },
}


class SVGModelMixin:
    # pylint:disable=missing-docstring,too-few-public-methods
    @staticmethod
    def _make_model(obs_space, action_space, config):
        config = deep_update(
            BASE_MODEL_CONFIG, config.get("model", {}), False, ["encoder"]
        )

        params_module = SVGDynamicsParams(obs_space, action_space, config)
        dist_module = Independent(Normal(), reinterpreted_batch_ndims=1)

        model = StochasticModel.assemble(params_module, dist_module, config)
        return {"model": model}


class SVGDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to distribution parameters through parallel
    subnetworks for each output dimension.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        obs_size, act_size = obs_space.shape[0], action_space.shape[0]

        def make_encoder():
            return StateActionEncoder(obs_size, act_size, **config["encoder"])

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


BASE_CONFIG = {
    "torch_script": False,
    "replay_kl": False,
    "actor": {
        "units": (100, 100),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    },
    "critic": {
        "target_vf": True,
        "encoder": {
            "units": (400, 200),
            "activation": "Tanh",
            "initializer_options": {"name": "xavier_uniform"},
        },
    },
}


class SVGModule(
    SVGModelMixin, StochasticActorMixin, StateValueMixin, AbstractModelActorCritic,
):
    """Module architecture with reparameterized actor and model.

    Allows inference of noise variables given existing samples.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic", "model"])
        super().__init__(obs_space, action_space, config)

        if config.get("replay_kl") is False:
            old = self._make_actor(obs_space, action_space, config)
            self.old_actor = old["actor"].requires_grad_(False)
