"""Actor-Critic architecture used in Trust-Region Policy Optimization (TRPO)."""
from typing import Dict

import gym.spaces as spaces
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

from raylab.utils.pytorch import initialize_
from .actor_critic import AbstractActorCritic
from .basic import FullyConnected
from .distributions import (
    ComposeTransform,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)
from .flows import Affine1DHalfFlow, ConditionalNormalizingFlow
from .state_value_mixin import StateValueMixin
from .stochastic_actor_mixin import StochasticActorMixin, StochasticPolicy


# Defaults for Hopper-v1
BASE_CONFIG = {
    "torch_script": True,
    "actor": {"num_flows": 4, "hidden_size": 3},
    "critic": {
        "encoder": {
            "units": (32, 32),
            "activation": "Tanh",
            "initializer_options": {"name": "normal", "std": 1.0},
        },
        "target_vf": False,
    },
}


class TRPOTang2018(StochasticActorMixin, StateValueMixin, AbstractActorCritic):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))

    @override(StochasticActorMixin)
    def _make_actor(self, obs_space, action_space, config):
        actor_config = config["actor"]
        return {"actor": NormalizingFlowsPolicy(obs_space, action_space, actor_config)}


class NormalizingFlowsPolicy(StochasticPolicy):
    """
    Stochastic policy architecture used in
    http://arxiv.org/abs/1809.10326
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        assert isinstance(
            action_space, spaces.Box
        ), f"Normalizing Flows incompatible with action space type {type(action_space)}"

        obs_size = obs_space.shape[0]
        self.act_size = act_size = action_space.shape[0]

        def make_mod(parity):
            nout = act_size // 2
            nin = act_size - nout
            if parity:
                nin, nout = nout, nin
            return MLP(nin, nout, hidden_size=config["hidden_size"])

        parities = [bool(i % 2) for i in range(config["num_flows"])]
        couplings = [Affine1DHalfFlow(p, make_mod(p), make_mod(p)) for p in parities]
        add_state = AddStateFlow(obs_size, act_size)
        squash = TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        )
        transforms = couplings[:1] + [add_state] + couplings[1:] + [squash]

        self.dist = TransformedDistribution(
            base_dist=Independent(Normal(), reinterpreted_batch_ndims=1),
            transform=ComposeTransform(transforms),
        )

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        shape = obs.shape[:-1] + (self.act_size,)
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape), "state": obs}


class AddStateFlow(ConditionalNormalizingFlow):
    """Incorporates state information by adding a state embedding."""

    def __init__(self, obs_size, act_size):
        super().__init__(event_dim=1)
        self.state_encoder = MLP(obs_size, act_size, hidden_size=64)

    @override(ConditionalNormalizingFlow)
    def _encode(self, inputs, cond: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(cond["state"])
        return inputs + encoded, torch.zeros(inputs.shape[: -self.event_dim])

    @override(ConditionalNormalizingFlow)
    def _decode(self, inputs, cond: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(cond["state"])
        return inputs - encoded, torch.zeros(inputs.shape[: -self.event_dim])


class MLP(nn.Module):
    # pylint:disable=missing-docstring
    def __init__(self, in_size, out_size, layer_norm=True, hidden_size=3):
        super().__init__()
        fully_connected = FullyConnected(
            in_size, units=(hidden_size,) * 2, activation="ReLU", layer_norm=layer_norm,
        )
        linear = nn.Linear(fully_connected.out_features, out_size)
        linear.apply(initialize_("uniform", a=-3e-3, b=3e-3))
        self.net = nn.Sequential(fully_connected, linear)

    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.net(inputs)
