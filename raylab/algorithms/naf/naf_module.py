"""Normalized Advantage Function nn.Module."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_orthogonal


class NAFModule(nn.Module):
    """Neural network module that implements the forward pass of NAF."""

    def __init__(self, obs_dim, action_low, action_high, config, script=None):
        super().__init__()
        self.logits_module = FullyConnectedModule(
            obs_dim, units=config["layers"], activation=config["activation"]
        )
        logit_dim = self.logits_module.out_features
        self.value_module = ValueModule(logit_dim, script=script)
        self.action_module = ActionModule(
            logit_dim, action_low, action_high, script=script
        )
        self.advantage_module = AdvantageModule(
            logit_dim, self.action_module.out_features, script=script
        )

        if script == "trace":
            fake_obs = torch.randn(1, obs_dim)
            fake_logits = torch.randn(1, logit_dim)
            fake_actions = torch.randn(1, self.action_module.out_features)
            self.logits_module = torch.jit.trace(self.logits_module, fake_obs)
            self.value_module = torch.jit.trace(self.value_module, fake_logits)
            self.action_module = torch.jit.trace(self.action_module, fake_logits)
            self.advantage_module = torch.jit.trace(
                self.advantage_module, (fake_logits, fake_actions, fake_actions)
            )
        elif script == "script":
            self.logits_module = torch.jit.script(self.logits_module)
            self.value_module = torch.jit.script(self.value_module)
            self.action_module = torch.jit.script(self.action_module)
            self.advantage_module = torch.jit.script(self.advantage_module)

        self.apply(initialize_orthogonal(config["ortho_init_gain"]))

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs)
        best_value = self.value_module(logits)
        best_action = self.action_module(logits)
        advantage = self.advantage_module(logits, best_action, actions)
        action_value = best_value + advantage
        return action_value, best_action, best_value


class ValueModule(nn.Module):
    """Neural network module implementing the value function term of NAF."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, script=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = 1
        self.linear_module = nn.Linear(self.in_features, self.out_features)

        if script == "trace":
            self.linear_module = torch.jit.trace(
                self.linear_module, torch.randn(1, self.in_features)
            )
        elif script == "script":
            self.linear_module = torch.jit.script(self.linear_module)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        return self.linear_module(logits)


class ActionModule(nn.Module):
    """Neural network module implementing the greedy action term of NAF."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, action_low, action_high, script=None):
        super().__init__()
        self.in_features = in_features
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_range", action_high - action_low)
        self.out_features = self.action_low.numel()
        self.linear_module = nn.Linear(self.in_features, self.out_features)

        if script == "trace":
            self.linear_module = torch.jit.trace(
                self.linear_module, torch.randn(1, self.in_features)
            )
        elif script == "script":
            self.linear_module = torch.jit.script(self.linear_module)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        unscaled_actions = self.linear_module(logits)
        squashed_actions = torch.sigmoid(unscaled_actions / self.action_range)
        scaled_actions = squashed_actions * self.action_range + self.action_low
        return scaled_actions


class AdvantageModule(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, action_dim, script=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = 1
        self.tril_module = TrilMatrixModule(self.in_features, action_dim, script=script)

        if script == "trace":
            self.tril_module = torch.jit.trace(
                self.tril_module, torch.randn(1, self.in_features)
            )
        elif script == "script":
            self.tril_module = torch.jit.script(self.tril_module)

    @override(nn.Module)
    def forward(self, logits, best_action, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_module(logits)  # square matrix [..., N, N]
        action_diff = (actions - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -0.5 * torch.norm(vec, p=2, dim=-2).pow(2)
        return advantage


class TrilMatrixModule(nn.Module):
    """Neural network module which outputs a lower-triangular matrix."""

    __constants__ = {"in_features", "matrix_dim"}

    def __init__(self, in_features, matrix_dim, script=None):
        super().__init__()
        self.in_features = in_features
        self.matrix_dim = matrix_dim
        tril_dim = int(self.matrix_dim * (self.matrix_dim + 1) / 2)
        self.linear_module = nn.Linear(self.in_features, tril_dim)

        if script == "trace":
            self.linear_module = torch.jit.trace(
                self.linear_module, torch.randn(1, self.in_features)
            )
        elif script == "script":
            self.linear_module = torch.jit.script(self.linear_module)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        # Batch of flattened lower triangular matrices: [..., N * (N + 1) / 2]
        flat_trils = self.linear_module(logits)
        # Batch of zero-valued matrices: [..., N, N]
        mats = torch.zeros(logits.shape[:1] + (self.matrix_dim, self.matrix_dim))
        # Mask of lower triangular indices: [N, N]
        mask = torch.tril(torch.ones(self.matrix_dim, self.matrix_dim))
        # Put flattened trils into appropriate indices of zeros
        mats[:, mask.to(torch.bool)] = flat_trils
        # Mask of diagonal indices: [N, N]
        diag = torch.diag(torch.ones(self.matrix_dim))
        # Triangular matrix with exponentiated diagonal
        tril = mats * mats * diag + (1 - diag) * mats
        return tril


class StateActionEncodingModule(nn.Module):
    """Neural network module which concatenates action after the first layer."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, obs_dim, action_dim, units=(), activation="relu"):
        super().__init__()
        self.in_features = obs_dim
        if units:
            self.obs_module = nn.Sequential(nn.Linear(obs_dim, units[0]), activation())
            input_dim = units[0] + action_dim
            units = units[1:]
            self.sequential_module = FullyConnectedModule(
                input_dim, units=units, activation=activation
            )
            self.out_features = self.sequential_module.out_features
        else:
            self.obs_module = nn.Identity()
            self.sequential_module = nn.Identity()
            self.out_features = obs_dim + action_dim

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        output = self.obs_module(obs)
        output = torch.cat([output, actions], dim=-1)
        output = self.sequential_module(output)
        return output


class FullyConnectedModule(nn.Module):
    """Neural network module that applies several fully connected modules to inputs."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, units=(), activation="relu"):
        super().__init__()
        self.in_features = in_features
        activation = get_activation(activation)
        units = [self.in_features] + units
        modules = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation())
        self.sequential_module = nn.Sequential(*modules)
        self.out_features = units[-1]

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.sequential_module(inputs)


def get_activation(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU
        if activation == "elu":
            return nn.ELU
        if activation == "tanh":
            return nn.Tanh
        raise NotImplementedError("Unsupported activation name '{}'".format(activation))
    raise ValueError(
        "'activation' must be a string type, got '{}'".format(type(activation))
    )
