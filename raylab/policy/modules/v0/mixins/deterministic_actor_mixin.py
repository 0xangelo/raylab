"""Support for modules with deterministic policies."""
import warnings

import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.pytorch.nn import FullyConnected
from raylab.pytorch.nn import GaussianNoise
from raylab.pytorch.nn import NormalizedLinear
from raylab.pytorch.nn import TanhSquash
from raylab.pytorch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge


BASE_CONFIG = {
    # === Twin Delayed DDPG (TD3) tricks ===
    # Add gaussian noise to the action when calculating the Deterministic
    # Policy Gradient
    "smooth_target_policy": True,
    # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
    "target_gaussian_sigma": 0.3,
    "separate_target_policy": False,
    "perturbed_policy": False,
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing
    # function
    "beta": 1.2,
    "initializer_options": {"name": "xavier_uniform"},
    "encoder": {"units": (32, 32), "activation": "ReLU", "layer_norm": False},
}


class DeterministicActorMixin:
    """Adds constructor for modules with deterministic policies."""

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config.get("actor", {}), False, ["encoder"])
        if not config["encoder"].get("layer_norm") and config["perturbed_policy"]:
            warnings.warn(
                "'layer_norm' is deactivated even though a perturbed policy was "
                "requested. For optimal stability, set 'layer_norm': True."
            )

        actor = DeterministicPolicy.from_scratch(obs_space, action_space, config)

        behavior = actor
        if config["perturbed_policy"]:
            behavior = DeterministicPolicy.from_scratch(obs_space, action_space, config)

        target_actor = actor
        if config["separate_target_policy"]:
            target_actor = DeterministicPolicy.from_scratch(
                obs_space, action_space, config
            )
            target_actor.load_state_dict(actor.state_dict())
        if config["smooth_target_policy"]:
            target_actor = DeterministicPolicy.from_existing(
                target_actor, noise=config["target_gaussian_sigma"],
            )

        return {"actor": actor, "behavior": behavior, "target_actor": target_actor}


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
        logits = FullyConnected(in_features=obs_space.shape[0], **config["encoder"])
        logits.apply(
            initialize_(
                activation=config["encoder"].get("activation"),
                **config["initializer_options"]
            )
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
