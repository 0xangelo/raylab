"""Normalized Advantage Function neural network modules."""
import warnings

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.dictionaries import deep_merge
from raylab.modules import FullyConnected, TrilMatrix, NormalizedLinear, TanhSquash
from .mixins import DeterministicPolicy


BASE_CONFIG = {
    # === Module Optimization ===
    # Whether to convert the module to a ScriptModule for faster inference
    "torch_script": False,
    "double_q": False,
    "encoder": {
        "units": (32, 32),
        "activation": "ELU",
        "initializer_options": {"name": "orthogonal"},
    },
    "perturbed_policy": False,
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing
    # function
    "beta": 1.2,
}


class NAFModule(nn.ModuleDict):
    """Module dict containing NAF's modules"""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        config = deep_merge(BASE_CONFIG, config, False, ["encoder"])
        self.update(self._make_critic(obs_space, action_space, config))
        self.update(self._make_actor(obs_space, action_space, config))

    def _make_critic(self, obs_space, action_space, config):
        logits = self._make_encoder(obs_space, config)
        naf = NAF(logits, action_space, config)
        critics = nn.ModuleList([naf])
        if config["double_q"]:
            twin_logits = self._make_encoder(obs_space, config)
            twin_naf = NAF(twin_logits, action_space, config)
            critics.append(twin_naf)

        def make_target():
            encoder = self._make_encoder(obs_space, config)
            return nn.Sequential(encoder, nn.Linear(encoder.out_features, 1))

        values = nn.ModuleList([nn.Sequential(m.logits, m.value) for m in critics])
        target_values = nn.ModuleList([make_target() for m in critics])
        return {
            "critics": critics,
            "vcritics": values,
            "target_vcritics": target_values,
        }

    @staticmethod
    def _make_encoder(obs_space, config):
        return FullyConnected(in_features=obs_space.shape[0], **config["encoder"])

    def _make_actor(self, obs_space, action_space, config):
        naf = self.critics[0]

        mods = nn.ModuleList([naf.logits, naf.pre_act, naf.squash])
        actor = DeterministicPolicy(mods)
        behavior = actor
        if config["perturbed_policy"]:
            if not config["encoder"].get("layer_norm"):
                warnings.warn(
                    "'layer_norm' is deactivated even though a perturbed policy was "
                    "requested. For optimal stability, set 'layer_norm': True."
                )
            behavior = DeterministicPolicy.from_scratch(obs_space, action_space, config)

        return {"actor": actor, "behavior": behavior}


class NAF(nn.Module):
    """Neural network module implementing the Normalized Advantage Function (NAF)."""

    def __init__(self, logits_module, action_space, config):
        super().__init__()
        self.logits = logits_module
        self.value = nn.Linear(self.logits.out_features, 1)
        self.pre_act = NormalizedLinear(
            in_features=self.logits.out_features,
            out_features=action_space.shape[0],
            beta=config["beta"],
        )
        self.squash = TanhSquash(
            torch.as_tensor(action_space.low), torch.as_tensor(action_space.high)
        )
        self.tril = TrilMatrix(self.logits.out_features, action_space.shape[0])
        self.advantage_module = AdvantageFunction(
            self.tril, nn.Sequential(self.pre_act, self.squash)
        )

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits(obs)
        best_value = self.value(logits)
        advantage = self.advantage_module(logits, actions)
        return advantage + best_value


class AdvantageFunction(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    def __init__(self, tril_module, action_module):
        super().__init__()
        self.tril_module = tril_module
        self.action_module = action_module

    @override(nn.Module)
    def forward(self, logits, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_module(logits)  # square matrix [..., N, N]
        best_action = self.action_module(logits)  # batch of actions [..., N]
        action_diff = (actions - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -0.5 * vec.transpose(-1, -2).matmul(vec).squeeze(-1)
        return advantage


class MultivariateNormalParams(nn.Module):
    """Neural network module implementing a multivariate gaussian policy."""

    def __init__(self, logits_module, loc_module, scale_tril_module):
        super().__init__()
        self.logits_module = logits_module
        self.loc_module = loc_module
        self.scale_tril_module = scale_tril_module

    @override(nn.Module)
    def forward(self, obs):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        loc = self.loc_module(logits)
        scale_tril = self.scale_tril_module(logits)
        return {"loc": loc, "scale_tril": scale_tril}
