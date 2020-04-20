"""Stochastic Model with Normalizing Flows density approximation."""
import warnings

import gym.spaces as spaces
from ray.rllib.utils.annotations import override
import torch.nn as nn

from raylab.utils.dictionaries import deep_merge
from ..basic import StateActionEncoder, NormalParams, StdNormalParams
from ..distributions import (
    CompositeTransform,
    Independent,
    Normal,
    TransformedDistribution,
)
from .. import flows
from .. import networks
from .stochastic_model_mixin import StochasticModel


BASE_CONFIG = {
    "residual": True,
    "conditional_prior": True,
    "input_encoder": {"units": (64, 64), "activation": "ReLU"},
    "num_flows": 4,
    "conditional_flow": False,
    "flow": {
        "type": "AffineCouplingTransform",
        "transform_net": {"type": "MLP", "num_blocks": 0},
    },
}


class NormalizingFlowModelMixin:
    """Stochastic model module with Normalizing Flow density estimator."""

    # pylint:disable=too-few-public-methods

    def _make_model(self, obs_space, action_space, config):
        config = deep_merge(
            BASE_CONFIG,
            config.get("model", {}),
            False,
            ["input_encoder", "flow"],
            ["flow"],
        )
        assert isinstance(
            obs_space, spaces.Box
        ), f"Normalizing Flow incompatible with observation space {type(obs_space)}"

        # PRIOR ========================================================================
        params_module, base_dist = self._make_model_prior(
            obs_space, action_space, config
        )
        # NormalizingFlow ==============================================================
        transforms = self._make_model_transforms(
            obs_space, params_module.state_size, config
        )
        dist_module = TransformedDistribution(
            base_dist=base_dist, transform=CompositeTransform(transforms),
        )

        return {"model": StochasticModel.assemble(params_module, dist_module, config)}

    @staticmethod
    def _make_model_prior(obs_space, action_space, config):
        # Ensure we're not encoding the inputs for nothing
        obs_size, act_size = obs_space.shape[0], action_space.shape[0]
        if config["conditional_prior"] or config["conditional_flow"]:
            input_encoder = StateActionEncoder(
                obs_size, act_size, **config["input_encoder"]
            )
        else:
            warnings.warn("Model is blind to the observations")
            input_encoder = StateActionEncoder(obs_size, act_size, units=())

        params_module = NFNormalParams(input_encoder, obs_space, config)
        base_dist = Independent(Normal(), reinterpreted_batch_ndims=1)
        return params_module, base_dist

    @staticmethod
    def _make_model_transforms(obs_space, state_size, config):
        obs_size = obs_space.shape[0]
        flow_config = config["flow"].copy()
        cls = getattr(flows, flow_config.pop("type"))

        if issubclass(cls, flows.CouplingTransform):
            net_config = flow_config.pop("transform_net")
            net_config.setdefault("hidden_features", obs_size)
            transform_net = getattr(networks, net_config.pop("type"))

            def transform_net_create_fn(in_features, out_features):
                return transform_net(
                    in_features,
                    out_features,
                    state_features=state_size if config["conditional_flow"] else None,
                    **net_config,
                )

            masks = [
                flows.masks.create_alternating_binary_mask(obs_size, bool(i % 2))
                for i in range(config["num_flows"])
            ]
            transforms = [cls(m, transform_net_create_fn, **flow_config) for m in masks]

        else:
            raise NotImplementedError(f"Unsupported flow type {cls}")

        return transforms


class NFNormalParams(nn.Module):
    """Maps inputs to distribution parameters for Normalizing Flows."""

    def __init__(self, input_encoder, obs_space, config):
        super().__init__()
        self.input_encoder = input_encoder
        self.state_size = self.input_encoder.out_features

        obs_size = obs_space.shape[0]
        if config["conditional_prior"]:
            self.params = NormalParams(self.state_size, obs_size)
        else:
            self.params = StdNormalParams(1, obs_size)

    @override(nn.Module)
    def forward(self, obs, act):  # pylint:disable=arguments-differ
        state = self.input_encoder(obs, act)
        params = self.params(state)
        # For use in conditional flows later
        params["state"] = state
        return params
