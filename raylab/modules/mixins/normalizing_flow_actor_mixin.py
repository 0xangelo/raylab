"""Stochastic Actor with Normalizing Flows density approximation."""
import warnings

import gym.spaces as spaces
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

from raylab.utils.dictionaries import deep_merge
from ..basic import FullyConnected, NormalParams, StdNormalParams
from ..distributions import (
    CompositeTransform,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)
from .. import flows
from .. import networks
from .stochastic_actor_mixin import StochasticPolicy


BASE_CONFIG = {
    "conditional_prior": True,
    "obs_encoder": {"units": (64, 64), "activation": "ReLU"},
    "num_flows": 4,
    "conditional_flow": False,
    "flow": {
        "type": "AffineCouplingTransform",
        "transform_net": {"type": "MLP", "num_blocks": 0},
    },
}


class NormalizingFlowActorMixin:
    """Stochastic actor module with Normalizing Flow density estimator."""

    # pylint:disable=too-few-public-methods

    def _make_actor(self, obs_space, action_space, config):
        config = deep_merge(
            BASE_CONFIG,
            config.get("actor", {}),
            False,
            ["obs_encoder", "flow"],
            ["flow"],
        )
        assert isinstance(
            action_space, spaces.Box
        ), f"Normalizing Flow incompatible with action space type {type(action_space)}"

        # PRIOR ========================================================================
        params_module, base_dist = self._make_actor_prior(
            obs_space, action_space, config
        )
        # NormalizingFlow ==============================================================
        transforms = self._make_actor_transforms(
            action_space, params_module.state_size, config
        )
        dist_module = TransformedDistribution(
            base_dist=base_dist, transform=CompositeTransform(transforms),
        )

        return {"actor": StochasticPolicy(params_module, dist_module)}

    @staticmethod
    def _make_actor_prior(obs_space, action_space, config):
        # Ensure we're not encoding the observation for nothing
        if config["conditional_prior"] or config["conditional_flow"]:
            obs_encoder = FullyConnected(obs_space.shape[0], **config["obs_encoder"])
        else:
            warnings.warn("Policy is blind to the observations")
            obs_encoder = FullyConnected(obs_space.shape[0], units=())

        params_module = NFNormalParams(obs_encoder, action_space, config)
        base_dist = Independent(Normal(), reinterpreted_batch_ndims=1)
        return params_module, base_dist

    @staticmethod
    def _make_actor_transforms(action_space, state_size, config):
        act_size = action_space.shape[0]
        flow_config = config["flow"].copy()
        cls = getattr(flows, flow_config.pop("type"))

        if issubclass(cls, flows.CouplingTransform):
            net_config = flow_config.pop("transform_net")
            net_config.setdefault("hidden_features", act_size)
            transform_net = getattr(networks, net_config.pop("type"))

            def transform_net_create_fn(in_features, out_features):
                return transform_net(
                    in_features,
                    out_features,
                    state_features=state_size if config["conditional_flow"] else None,
                    **net_config,
                )

            masks = [
                flows.masks.create_alternating_binary_mask(act_size, bool(i % 2))
                for i in range(config["num_flows"])
            ]
            transforms = [cls(m, transform_net_create_fn, **flow_config) for m in masks]

        else:
            raise NotImplementedError(f"Unsupported flow type {cls}")

        squash = TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        )
        return transforms + [squash]


class NFNormalParams(nn.Module):
    """Maps inputs to distribution parameters for Normalizing Flows."""

    def __init__(self, obs_encoder, action_space, config):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.state_size = self.obs_encoder.out_features

        act_size = action_space.shape[0]
        if config["conditional_prior"]:
            self.params = NormalParams(self.state_size, act_size)
        else:
            self.params = StdNormalParams(1, act_size)

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        state = self.obs_encoder(obs)
        params = self.params(state)
        # For use in conditional flows later
        params["state"] = state
        return params
