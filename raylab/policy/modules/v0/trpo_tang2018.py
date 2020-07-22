"""Actor-Critic architecture used in Trust-Region Policy Optimization (TRPO)."""
from typing import Dict

import gym.spaces as spaces
import torch
import torch.nn as nn
from ray.rllib.utils import merge_dicts
from ray.rllib.utils import override

import raylab.policy.modules.networks as networks
from raylab.pytorch.nn import FullyConnected
from raylab.pytorch.nn.distributions import flows
from raylab.pytorch.nn.distributions import Independent
from raylab.pytorch.nn.distributions import Normal
from raylab.pytorch.nn.distributions import TransformedDistribution
from raylab.pytorch.nn.distributions.flows import CompositeTransform
from raylab.pytorch.nn.distributions.flows import TanhSquashTransform
from raylab.pytorch.nn.init import initialize_

from .abstract import AbstractActorCritic
from .mixins import StateValueMixin
from .mixins import StochasticActorMixin
from .mixins import StochasticPolicy


# Defaults for Hopper-v1
BASE_CONFIG = {
    "actor": {"num_flows": 4, "hidden_size": 3},
    "critic": {
        "initializer_options": {"name": "normal", "std": 1.0},
        "encoder": {"units": (32, 32), "activation": "Tanh"},
        "target_vf": False,
    },
}


class TRPOTang2018(StochasticActorMixin, StateValueMixin, AbstractActorCritic):
    """Actor-Critic module with stochastic actor and state-value critics.

    Stochastic policy architecture used in
    http://arxiv.org/abs/1809.10326
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))

    @override(StochasticActorMixin)
    def _make_actor(self, obs_space, action_space, config):
        config = config["actor"]
        assert isinstance(
            action_space, spaces.Box
        ), f"Normalizing Flows incompatible with action space type {type(action_space)}"

        # PARAMS MODULE ================================================================
        params_module = StateNormalParams(obs_space, action_space)

        # FLOW MODULES =================================================================
        obs_size = obs_space.shape[0]
        act_size = action_space.shape[0]

        def transform_net_fn(in_features, out_features):
            return networks.MLP(
                in_features,
                out_features,
                hidden_features=config["hidden_size"],
                num_blocks=2,
                activation="ReLU",
            )

        masks = [
            flows.masks.create_alternating_binary_mask(act_size, bool(i % 2))
            for i in range(config["num_flows"])
        ]
        couplings = [flows.AffineCouplingTransform(m, transform_net_fn) for m in masks]
        add_state = AddStateFlow(obs_size, act_size)
        squash = TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        )
        transforms = couplings[:1] + [add_state] + couplings[1:] + [squash]
        dist_module = TransformedDistribution(
            base_dist=Independent(Normal(), reinterpreted_batch_ndims=1),
            transform=CompositeTransform(transforms),
        )

        return {"actor": StochasticPolicy(params_module, dist_module)}


class StateNormalParams(nn.Module):
    """Maps observations to standard normal parameters and forwards observations."""

    def __init__(self, obs_space, action_space):
        super().__init__()
        self.obs_dim = len(obs_space.shape)
        self.act_shape = action_space.shape

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        shape = obs.shape[: -self.obs_dim] + self.act_shape
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape), "state": obs}


class AddStateFlow(flows.ConditionalTransform):
    """Incorporates state information by adding a state embedding."""

    def __init__(self, obs_size, act_size):
        super().__init__(event_dim=1)
        self.state_encoder = MLP(obs_size, act_size, hidden_size=64)

    @override(flows.ConditionalTransform)
    def encode(self, inputs, params: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(params["state"])
        return inputs + encoded, torch.zeros(inputs.shape[: -self.event_dim])

    @override(flows.ConditionalTransform)
    def decode(self, inputs, params: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(params["state"])
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
