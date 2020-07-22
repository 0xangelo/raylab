"""SVG model architecture."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.pytorch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge

from .stochastic_model_mixin import StochasticModel


BASE_CONFIG = {
    "residual": True,
    "input_dependent_scale": False,
    "encoder": {"units": (40, 40), "activation": "Tanh", "delay_action": True},
    "initializer_options": {"name": "xavier_uniform"},
}


class SVGModelMixin:
    # pylint:disable=missing-docstring,too-few-public-methods
    @staticmethod
    def _make_model(obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config.get("model", {}), False, ["encoder"])

        params_module = SVGDynamicsParams(obs_space, action_space, config)
        dist_module = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)

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
            mlp = nnx.StateActionEncoder(obs_size, act_size, **config["encoder"])
            mlp.apply(
                initialize_(
                    activation=config["encoder"].get("activation"),
                    **config["initializer_options"],
                )
            )
            return mlp

        self.logits = nn.ModuleList([make_encoder() for _ in range(obs_space.shape[0])])

        def make_param(in_features):
            kwargs = dict(event_size=1, input_dependent_scale=False)
            return nnx.NormalParams(in_features, **kwargs)

        self.params = nn.ModuleList([make_param(m.out_features) for m in self.logits])

    @override(nn.Module)
    def forward(self, obs, act):  # pylint:disable=arguments-differ
        params = [p(l(obs, act)) for p, l in zip(self.params, self.logits)]
        loc = torch.cat([d["loc"] for d in params], dim=-1)
        scale = torch.cat([d["scale"] for d in params], dim=-1)
        return {"loc": loc, "scale": scale}
