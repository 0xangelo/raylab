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


BASE_CONFIG = {
    "torch_script": True,
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
        "target_vf": False,
    },
}


class TRPOTang2018(
    StochasticActorMixin, StateValueMixin, AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))

    @override(StochasticActorMixin)
    def _make_actor(self, obs_space, action_space, config):
        actor_config = config["actor"]
        return {"actor": NormalizingFlowsPolicy(obs_space, action_space, actor_config)}

    @override(StateValueMixin)
    def _make_critic(self, obs_space, action_space, config):
        critic_config = config["critic"]
        critic_config["units"] = (64, 64)
        critic_config["activation"] = "ReLU"
        critic_config["target_vf"] = False
        return super()._make_critic(obs_space, action_space, config)


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
            mlp = FullyConnected(
                nin, units=(3,) * 4, activation="ELU", **config["initializer_options"],
            )
            linear = nn.Linear(mlp.out_features, nout)
            linear.apply(initialize_("orthogonal", gain=0.01))
            return nn.Sequential(mlp, linear)

        parities = [bool(i % 2) for i in range(4)]
        couplings = [Affine1DHalfFlow(p, make_mod(p), make_mod(p)) for p in parities]
        add_state = AddStateFlow(obs_size, act_size, config)
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

    def __init__(self, obs_size, act_size, config):
        super().__init__(event_dim=1)
        mlp = FullyConnected(
            obs_size,
            units=(64, 64),
            activation=config["activation"],
            **config["initializer_options"],
        )
        linear = nn.Linear(mlp.out_features, act_size)
        linear.apply(initialize_("orthogonal", gain=0.01))
        self.state_encoder = nn.Sequential(mlp, linear)

    @override(ConditionalNormalizingFlow)
    def _encode(self, inputs, cond: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(cond["state"])
        return inputs + encoded, torch.zeros(inputs.shape[: -self.event_dim])

    @override(ConditionalNormalizingFlow)
    def _decode(self, inputs, cond: Dict[str, torch.Tensor]):
        encoded = self.state_encoder(cond["state"])
        return inputs - encoded, torch.zeros(inputs.shape[: -self.event_dim])
