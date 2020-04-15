"""Stochastic Actor with Normalizing Flows density approximation."""
from typing import Dict

import gym.spaces as spaces
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

from raylab.utils.pytorch import initialize_
from ..basic import FullyConnected, NormalParams
from ..distributions import (
    ComposeTransform,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)
from ..flows import Affine1DHalfFlow, CondAffine1DHalfFlow
from .stochastic_actor_mixin import StochasticPolicy


BASE_CONFIG = {
    "obs_dependent_prior": True,
    "obs_encoder": {"units": (64, 64), "activation": "ELU"},
    "num_flows": 4,
    "state_cond_flow": False,
    "flow_mlp": {"units": (24,) * 4, "activation": "ELU"},
}


class NormalizingFlowActorMixin:
    """Stochastic actor module with Normalizing Flow density estimator."""

    # pylint:disable=too-few-public-methods

    def _make_actor(self, obs_space, action_space, config):
        config = deep_update(
            BASE_CONFIG, config.get("actor", {}), False, ["obs_encoder", "flow_mlp"]
        )
        assert isinstance(
            action_space, spaces.Box
        ), f"Normalizing Flow incompatible with action space type {type(action_space)}"

        # PRIOR ========================================================================
        params_module, base_dist = self._make_prior(obs_space, action_space, config)

        # NormalizingFlow ==============================================================
        transforms = self._make_transforms(
            action_space, params_module.state_size, config
        )
        dist_module = TransformedDistribution(
            base_dist=base_dist, transform=ComposeTransform(transforms),
        )

        return {"actor": StochasticPolicy(params_module, dist_module)}

    @staticmethod
    def _make_prior(obs_space, action_space, config):
        # Ensure we're not encoding the observation for nothing
        if config["obs_dependent_prior"] or config["state_cond_flow"]:
            obs_encoder = FullyConnected(obs_space.shape[0], **config["obs_encoder"])
        else:
            # WARNING: This effectively means the policy is blind to the observations
            obs_encoder = FullyConnected(obs_space.shape[0], units=())
        params_module = NFNormalParams(obs_encoder, action_space, config)
        base_dist = Independent(Normal(), reinterpreted_batch_ndims=1)
        return params_module, base_dist

    @staticmethod
    def _make_transforms(action_space, state_size, config):
        def transform_net(in_size, out_size):
            logits = FullyConnected(in_size, **config["flow_mlp"])
            linear = nn.Linear(logits.out_features, out_size)
            linear.apply(initialize_("orthogonal", gain=0.1))
            return nn.Sequential(logits, linear)

        act_size = action_space.shape[0]

        def make_mod(parity):
            in_size = (act_size + 1) // 2
            out_size = act_size // 2
            if parity:
                in_size, out_size = out_size, in_size

            if config["state_cond_flow"]:
                return CombineAndForward(
                    in_size, state_size, lambda s: transform_net(s, out_size)
                )
            return transform_net(in_size, out_size)

        flow_cls = (
            CondAffine1DHalfFlow if config["state_cond_flow"] else Affine1DHalfFlow
        )
        parities = [bool(i % 2) for i in range(config["num_flows"])]
        couplings = [flow_cls(p, make_mod(p), make_mod(p)) for p in parities]
        squash = TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        )
        transforms = couplings + [squash]
        return transforms


class CombineAndForward(nn.Module):
    """
    Combine inputs with 'state' value from dict and forward result to submodule.
    """

    def __init__(self, in_size, state_size, module_fn):
        super().__init__()
        combined_size = in_size + state_size
        self.module = module_fn(combined_size)

    @override(nn.Module)
    def forward(self, inputs, params: Dict[str, torch.Tensor]):
        # pylint:disable=arguments-differ
        # Expand in case sample_shape in base dist is non-singleton
        state = params["state"].expand(inputs.shape[:-1] + params["state"].shape[-1:])
        out = torch.cat([inputs, state], dim=-1)
        return self.module(out)


class NFNormalParams(nn.Module):
    """Maps inputs to distribution parameters for Normalizing Flows."""

    def __init__(self, obs_encoder, action_space, config):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.state_size = self.obs_encoder.out_features

        act_size = action_space.shape[0]
        if config["obs_dependent_prior"]:
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


class StdNormalParams(nn.Module):
    """Produces standard Normal parameters and expands on input."""

    def __init__(self, input_dim, event_size):
        super().__init__()
        self.input_dim = input_dim
        self.event_shape = (event_size,)

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        shape = inputs.shape[: -self.input_dim] + self.event_shape
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape)}
