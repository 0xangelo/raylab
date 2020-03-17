"""Normalized Advantage Function neural network modules."""
import torch
import torch.nn as nn
import torch.distributions as dists
from ray.rllib.utils.annotations import override

from raylab.modules import (
    FullyConnected,
    TrilMatrix,
    NormalizedLinear,
    TanhSquash,
    GaussianNoise,
    DistRSample,
)


class NAFModule(nn.ModuleDict):
    """Module dict containing NAF's modules"""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self._script = config["torch_script"]
        self.update(self._make_naf(obs_space, action_space, config))
        self.target_value = nn.ModuleList(
            [self._make_value(obs_space, config) for _ in self.value]
        )
        self.target_value.load_state_dict(self.value.state_dict())
        self.policy = nn.Sequential(self.logits, self.mu, self.squash)
        if self._script:
            self.policy = torch.jit.script(self.policy)
        self.update(self._make_sampler(obs_space, action_space, config))

    def _make_naf(self, obs_space, action_space, config):
        modules = self._make_components(obs_space, action_space, config)
        naf_module = NAF(
            modules["logits"],
            modules["value"],
            modules["tril"],
            nn.Sequential(modules["mu"], modules["squash"]),
        )
        modules["naf"] = nn.ModuleList([naf_module])
        modules["value"] = nn.ModuleList(
            [nn.Sequential(modules["logits"], modules["value"])]
        )
        if config["clipped_double_q"]:
            twin_modules = self._make_components(obs_space, action_space, config)
            twin_naf = NAF(
                twin_modules["logits"],
                twin_modules["value"],
                twin_modules["tril"],
                nn.Sequential(twin_modules["mu"], twin_modules["squash"]),
            )
            modules["naf"].append(twin_naf)
            modules["value"].append(
                nn.Sequential(twin_modules["logits"], twin_modules["value"])
            )

        if self._script:
            modules["naf"] = nn.ModuleList(
                [m.as_script_module() for m in modules["naf"]]
            )
            modules["value"] = nn.ModuleList(
                [torch.jit.script(m) for m in modules["value"]]
            )
        return modules

    def _make_value(self, obs_space, config):
        logits = self._make_encoder(obs_space, config)
        val = nn.Linear(logits.out_features, 1)
        value = nn.Sequential(logits, val.as_script_module() if self._script else val)
        return torch.jit.script(value) if self._script else value

    def _make_components(self, obs_space, action_space, config):
        logits = self._make_encoder(obs_space, config)
        components = {
            "logits": logits,
            "value": nn.Linear(logits.out_features, 1),
            "mu": NormalizedLinear(
                in_features=logits.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            ),
            "squash": TanhSquash(
                torch.as_tensor(action_space.low), torch.as_tensor(action_space.high)
            ),
            "tril": TrilMatrix(logits.out_features, action_space.shape[0]),
        }
        if self._script:
            for key in ("value", "mu", "squash", "tril"):
                components[key] = components[key].as_script_module()
        return components

    def _make_sampler(self, obs_space, action_space, config):
        # Configure sampler module based on exploration strategy
        if config["exploration"] == "full_gaussian":
            sampler = MultivariateNormalSampler(
                action_space, self.logits, self.mu, self.tril
            )
            return {"sampler": sampler.as_script_module() if self._script else sampler}
        if config["exploration"] == "parameter_noise":
            logits_module = self._make_encoder(obs_space, config)
            sampler = nn.Sequential(
                logits_module,
                NormalizedLinear(
                    in_features=logits_module.out_features,
                    out_features=action_space.shape[0],
                    beta=config["beta"],
                ),
                TanhSquash(
                    torch.as_tensor(action_space.low),
                    torch.as_tensor(action_space.high),
                ),
            )
            sampler = torch.jit.script(sampler) if self._script else sampler
            return {"sampler": sampler, "perturbed_policy": sampler}
        if config["exploration"] == "diag_gaussian":
            expl_noise = GaussianNoise(config["diag_gaussian_stddev"])
            sampler = nn.Sequential(self.logits, self.mu, expl_noise, self.squash)
            sampler = torch.jit.script(sampler) if self._script else sampler
            return {"sampler": sampler}
        return {"sampler": self.policy}

    @staticmethod
    def _make_encoder(obs_space, config):
        module = FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            layer_norm=config.get(
                "layer_norm", config["exploration"] == "parameter_noise"
            ),
            **config["initializer_options"],
        )
        return module.as_script_module() if config["torch_script"] else module


class MultivariateNormalSampler(nn.Sequential):
    """Neural network module mapping inputs to MultivariateNormal parameters.

    This module is initialized to be close to a standard Normal distribution.
    """

    def __init__(self, action_space, logits_mod, mu_mod, tril_mod):
        params_module = MultivariateNormalParams(logits_mod, mu_mod, tril_mod)
        rsample_module = DistRSample(
            dists.MultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        super().__init__(params_module, rsample_module)

    def as_script_module(self):
        """Return self as a ScriptModule."""
        self[0] = self[0].as_script_module()
        inputs = {
            "loc": torch.randn_like(self[1].low).unsqueeze(0),
            "scale_tril": torch.diag(torch.ones_like(self[1].low)).unsqueeze(0),
        }
        self[1] = torch.jit.trace(self[1], inputs, check_trace=False)
        return torch.jit.script(self)


class NAF(nn.Module):
    """Neural network module implementing the Normalized Advantage Function (NAF)."""

    def __init__(self, logits_module, value_module, tril_module, action_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module
        self.advantage_module = AdvantageFunction(tril_module, action_module)

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        best_value = self.value_module(logits)
        advantage = self.advantage_module(logits, actions)
        return advantage + best_value

    def as_script_module(self):
        """Return self as a ScriptModule."""
        self.advantage_module = self.advantage_module.as_script_module()
        return torch.jit.script(self)


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

    def as_script_module(self):
        """Return self as a ScriptModule."""
        return torch.jit.script(self)


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

    def as_script_module(self):
        """Return self as a ScriptModule."""
        return torch.jit.script(self)
