"""Actor-Critic architecture with deterministic actor."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.modules import (
    FullyConnected,
    NormalizedLinear,
    TanhSquash,
    GaussianNoise,
    StateActionEncoder,
)


class DeterministicActorCritic(nn.ModuleDict):
    """Module containing deterministic policy and action value functions."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_actor(obs_space, action_space, config))
        self.update(self._make_critic(obs_space, action_space, config))

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        modules = {}
        policy_config = config["policy"]
        if "layer_norm" not in policy_config:
            policy_config["layer_norm"] = config["exploration"] == "parameter_noise"

        modules["policy"] = DeterministicPolicy.from_scratch(
            obs_space, action_space, policy_config
        )

        if config["exploration"] == "gaussian":
            modules["sampler"] = DeterministicPolicy.from_existing(
                modules["policy"], noise=config["exploration_gaussian_sigma"],
            )
        elif config["exploration"] == "parameter_noise":
            modules["sampler"] = DeterministicPolicy.from_scratch(
                obs_space, action_space, policy_config
            )
        else:
            modules["sampler"] = modules["policy"]

        if config["smooth_target_policy"]:
            modules["target_policy"] = DeterministicPolicy.from_existing(
                modules["policy"], noise=config["target_gaussian_sigma"],
            )
        else:
            modules["target_policy"] = modules["policy"]

        return modules

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        critic_config = config["critic"]

        def make_critic():
            return ActionValueFunction.from_scratch(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                delay_action=critic_config["delay_action"],
                units=critic_config["units"],
                activation=critic_config["activation"],
                **critic_config["initializer_options"]
            )

        n_critics = 2 if config["double_q"] else 1
        critics = nn.ModuleList([make_critic() for _ in range(n_critics)])
        target_critics = nn.ModuleList([make_critic() for _ in range(n_critics)])
        target_critics.load_state_dict(critics.state_dict())
        return {"critics": critics, "target_critics": target_critics}


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
        mods = nn.ModuleList(
            [
                logits,
                NormalizedLinear(
                    in_features=logits.out_features,
                    out_features=action_space.shape[0],
                    beta=config["beta"],
                ),
                TanhSquash(
                    torch.as_tensor(action_space.low),
                    torch.as_tensor(action_space.high),
                ),
            ]
        )
        return cls(mods, noise=noise)

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.sequential(inputs)


class ActionValueFunction(nn.Module):
    """Neural network module emulating a Q value function."""

    def __init__(self, logits_module, value_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs, actions)
        return self.value_module(logits)

    @classmethod
    def from_scratch(cls, *logits_args, **logits_kwargs):
        """Create an action value function with new logits and value modules."""
        logits_module = StateActionEncoder(*logits_args, **logits_kwargs)
        value_module = nn.Linear(logits_module.out_features, 1)
        return cls(logits_module, value_module)
