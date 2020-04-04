""" Trust-Region Policy Optimization with RealNVP density approximation."""
from typing import Dict

import gym.spaces as spaces
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

from raylab.utils.pytorch import initialize_
from .actor_critic import AbstractActorCritic
from .basic import FullyConnected, StateActionEncoder
from .distributions import (
    ComposeTransform,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)
from .flows import CondAffine1DHalfFlow
from .state_value_mixin import StateValueMixin
from .stochastic_actor_mixin import StochasticActorMixin, StochasticPolicy


BASE_CONFIG = {
    "torch_script": True,
    "actor": {
        "units": (64, 64),
        "activation": "ELU",
        "initializer_options": {"name": "xavier_uniform"},
        "num_flows": 4,
        "flow": {
            "units": (24,) * 4,
            "activation": "ELU",
            "initializer_options": {"name": "xavier_uniform"},
        },
    },
    "critic": {
        "units": (64, 64),
        "activation": "ELU",
        "initializer_options": {"name": "xavier_uniform"},
        "target_vf": False,
    },
}


class TRPORealNVP(
    StochasticActorMixin, StateValueMixin, AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))

    @override(StochasticActorMixin)
    def _make_actor(self, obs_space, action_space, config):
        actor_config = config["actor"]
        return {"actor": RealNVPPolicy(obs_space, action_space, actor_config)}


class CondMLP(nn.Module):
    # pylint:disable=missing-docstring

    def __init__(self, parity, action_size, state_size, **mlp_kwargs):
        super().__init__()
        out_size = action_size // 2
        in_size = action_size - out_size
        if parity:
            in_size, out_size = out_size, in_size
        self.encoder = StateActionEncoder(
            in_size,
            state_size,
            delay_action=False,
            units=mlp_kwargs["units"],
            activation=mlp_kwargs["activation"],
            **mlp_kwargs["initializer_options"],
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
        ), f"RealNVPPolicy incompatible with action space type {type(action_space)}"

        obs_size = obs_space.shape[0]
        self.act_size = act_size = action_space.shape[0]

        # STATE ENCODER ========================================================
        self.obs_encoder = FullyConnected(
            obs_size,
            units=config["units"],
            activation=config["activation"],
            **config["initializer_options"],
        )

        # RealNVP ==============================================================
        flow_config = config["flow"]

        def make_mod(parity):
            return CondMLP(
                parity, act_size, self.obs_encoder.out_features, **flow_config
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
