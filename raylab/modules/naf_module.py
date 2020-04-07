"""Normalized Advantage Function neural network modules."""
import torch
import torch.nn as nn
import torch.distributions as dists
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override

from raylab.modules import (
    FullyConnected,
    TrilMatrix,
    NormalizedLinear,
    TanhSquash,
    DistRSample,
)


BASE_CONFIG = {
    "double_q": False,
    "units": (32, 32),
    "activation": "ELU",
    "initializer_options": {"name": "orthogonal"},
    "perturbed_policy": False,
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing
    # function
    "beta": 1.2,
    # === Module Optimization ===
    # Whether to convert the module to a ScriptModule for faster inference
    "torch_script": False,
}


class NAFModule(nn.ModuleDict):
    """Module dict containing NAF's modules"""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        config = merge_dicts(BASE_CONFIG, config)
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

        target_critics = nn.ModuleList([make_target() for m in critics])
        return {"critics": critics, "target_critics": target_critics}

    def _make_actor(self, obs_space, action_space, config):
        naf = self.critics[0]
        actor = nn.ModuleDict()
        actor.policy = nn.Sequential(naf.logits, naf.pre_act, naf.squash)
        if config["perturbed_policy"]:
            encoder = self._make_encoder(obs_space, config)
            pre_act = NormalizedLinear(
                in_features=encoder.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            )
            squash = TanhSquash(
                torch.as_tensor(action_space.low), torch.as_tensor(action_space.high)
            )
            actor.behavior = nn.Sequential(encoder, pre_act, squash)

        return {"actor": actor}

    @staticmethod
    def _make_encoder(obs_space, config):
        return FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            layer_norm=config.get("layer_norm", config["perturbed_policy"]),
            **config["initializer_options"],
        )


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


class MultivariateNormalSampler(nn.Sequential):
    """Neural network module mapping inputs to MultivariateNormal parameters.

    This module is initialized to be close to a standard Normal distribution.
    """

    def __init__(self, action_space, logits_mod, mu_mod, tril_mod, *, script=False):
        params_module = MultivariateNormalParams(logits_mod, mu_mod, tril_mod)
        rsample_module = DistRSample(
            dists.MultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )

        if script:
            act_dim = action_space.shape[0]
            rsample_module = rsample_module.traced(
                {
                    "loc": torch.zeros(1, act_dim),
                    "scale_tril": torch.eye(act_dim).unsqueeze(0),
                }
            )
        super().__init__(params_module, rsample_module)


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
