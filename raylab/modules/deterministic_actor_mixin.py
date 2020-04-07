"""Support for modules with deterministic policies."""
import warnings

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.modules import (
    FullyConnected,
    NormalizedLinear,
    TanhSquash,
    GaussianNoise,
)


class DeterministicActorMixin:
    """Adds constructor for modules with deterministic policies."""

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        actor = nn.ModuleDict()
        actor_config = config["actor"]
        if "layer_norm" not in actor_config and config["perturbed_policy"]:
            warnings.warn(
                "'layer_norm' is deactivated even though a perturbed policy was "
                "requested. For optimal stability, set 'layer_norm': True."
            )

        actor.policy = DeterministicPolicy.from_scratch(
            obs_space, action_space, actor_config
        )
        if config["perturbed_policy"]:
            actor.behavior = DeterministicPolicy.from_scratch(
                obs_space, action_space, actor_config
            )
        else:
            actor.behavior = actor.policy

        if config["smooth_target_policy"]:
            actor.target_policy = DeterministicPolicy.from_existing(
                actor.policy, noise=config["target_gaussian_sigma"],
            )
        else:
            actor.target_policy = actor.policy

        return {"actor": actor}


class DeterministicPolicy(nn.Module):
    """Represents a deterministic policy as a sequence of modules.

    If a noise param is passed, adds Gaussian white noise before the squashing module.
    """

    def __init__(self, mods, *, noise=None):
        super().__init__()
        if noise:
            noise_mod = nn.ModuleList([GaussianNoise(noise)])
            mods = mods[:-1].extend(noise_mod).extend(mods[-1:])
        self.mods = mods
        self.sequential = nn.Sequential(*self.mods)

    @classmethod
    def from_existing(cls, policy, *, noise=None):
        """Create a policy using an existing's modules."""
        mods = policy.mods
        if len(mods) > 3:
            mods = mods[:2].extend(mods[-1:])
        return cls(mods, noise=noise)

    @classmethod
    def from_scratch(cls, obs_space, action_space, config, *, noise=None):
        """Create a policy using new modules."""
        logits = FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            layer_norm=config["layer_norm"],
            **config["initializer_options"]
        )
        normalize = NormalizedLinear(
            in_features=logits.out_features,
            out_features=action_space.shape[0],
            beta=config["beta"],
        )
        squash = TanhSquash(
            torch.as_tensor(action_space.low), torch.as_tensor(action_space.high),
        )
        mods = nn.ModuleList([logits, normalize, squash])
        return cls(mods, noise=noise)

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        return self.sequential(obs)
