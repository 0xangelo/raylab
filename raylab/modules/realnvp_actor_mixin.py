"""Trust-Region Policy Optimization with RealNVP density approximation."""
from typing import Dict

import gym.spaces as spaces
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

from raylab.utils.pytorch import initialize_
from .basic import FullyConnected, StateActionEncoder
from .distributions import (
    ComposeTransform,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)
from .flows import CondAffine1DHalfFlow
from .stochastic_actor_mixin import StochasticPolicy


BASE_CONFIG = {
    "obs_encoder": {
        "units": (64, 64),
        "activation": "ELU",
        "layer_norm": False,
        "initializer_options": {"name": "xavier_uniform"},
    },
    "num_flows": 4,
    "flow_mlp": {
        "units": (24,) * 4,
        "activation": "ELU",
        "layer_norm": False,
        "initializer_options": {"name": "xavier_uniform"},
    },
}


class RealNVPActorMixin:
    """Stochastic actor module with RealNVP density estimator."""

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        config = deep_update(
            BASE_CONFIG, config["actor"], False, ["obs_encoder", "flow_mlp"]
        )
        return {"actor": RealNVPPolicy(obs_space, action_space, config)}


class CondMLP(nn.Module):
    # pylint:disable=missing-docstring

    def __init__(self, parity, action_size, state_size, config):
        super().__init__()
        in_size = (action_size + 1) // 2
        out_size = action_size // 2
        if parity:
            in_size, out_size = out_size, in_size

        self.encoder = StateActionEncoder(
            in_size, state_size, delay_action=False, **config
        )
        self.linear = nn.Linear(self.encoder.out_features, out_size)
        self.linear.apply(initialize_("orthogonal", gain=0.01))

    def forward(self, inputs, cond: Dict[str, torch.Tensor]):
        # pylint:disable=arguments-differ
        state = cond["state"].expand(inputs.shape[:-1] + cond["state"].shape[-1:])
        return self.linear(self.encoder(inputs, state))


class RealNVPPolicy(StochasticPolicy):
    """Stochastic policy architecture with RealNVP density."""

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        assert isinstance(
            action_space, spaces.Box
        ), f"RealNVPPolicy is incompatible with action space type {type(action_space)}"

        obs_size = obs_space.shape[0]
        self.act_size = act_size = action_space.shape[0]

        # OBSERVATION ENCODER ========================================================
        self.obs_encoder = FullyConnected(obs_size, **config["obs_encoder"])

        # RealNVP ====================================================================
        def make_mod(parity):
            return CondMLP(
                parity, act_size, self.obs_encoder.out_features, config["flow_mlp"]
            )

        parities = [bool(i % 2) for i in range(config["num_flows"])]
        couplings = [
            CondAffine1DHalfFlow(p, make_mod(p), make_mod(p)) for p in parities
        ]
        squash = TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        )
        transforms = couplings + [squash]
        self.dist = TransformedDistribution(
            base_dist=Independent(Normal(), reinterpreted_batch_ndims=1),
            transform=ComposeTransform(transforms),
        )

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        shape = obs.shape[:-1] + (self.act_size,)
        state = self.obs_encoder(obs)
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape), "state": state}