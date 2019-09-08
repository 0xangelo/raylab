"""Normalized Advantage Function nn.Module."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_orthogonal


class NAFModule(nn.Module):
    """Neural network module that implements the forward pass of NAF."""

    def __init__(self, obs_dim, action_low, action_high, config):
        super().__init__()
        layers = config["layers"]
        self.logits_module = FullyConnectedModule(
            obs_dim, units=layers, activation=config["activation"]
        )

        logit_dim = layers[-1] if layers else obs_dim
        action_dim = action_low.numel()

        self.value_module = ValueModule(logit_dim)
        self.action_module = ActionModule(logit_dim, action_low, action_high)
        self.advantage_module = AdvantageModule(logit_dim, action_dim)

        self.logits_module.apply(initialize_orthogonal(config["ortho_init_gain"]))
        self.value_module.apply(initialize_orthogonal(0.01))
        self.action_module.apply(initialize_orthogonal(0.01))
        self.advantage_module.apply(initialize_orthogonal(0.01))

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

    def __init__(self, logit_dim):
        super().__init__()
        self.linear_module = nn.Linear(logit_dim, 1)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        return self.linear_module(logits)


class ActionModule(nn.Module):
    """Neural network module implementing the greedy action term of NAF."""

    def __init__(self, logit_dim, action_low, action_high):
        super().__init__()
        self.action_low = action_low
        self.action_range = action_high - action_low
        action_dim = self.action_low.numel()
        self.linear_module = nn.Linear(logit_dim, action_dim)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        unscaled_actions = self.linear_module(logits)
        squashed_actions = torch.sigmoid(unscaled_actions / self.action_range)
        scaled_actions = squashed_actions * self.action_range + self.action_low
        return scaled_actions


class AdvantageModule(nn.Module):
    """Neural network module implementing the advantage function term of NAF."""

    def __init__(self, logit_dim, action_dim):
        super().__init__()
        self.tril_matrix_module = TrilMatrixModule(logit_dim, action_dim)

    @override(nn.Module)
    def forward(self, logits, best_action, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_matrix_module(logits)  # square matrix [..., N, N]
        action_diff = (actions - best_action).unsqueeze(-1)  # column vector [..., N, 1]
        vec = tril_matrix.matmul(action_diff)  # column vector [..., N, 1]
        advantage = -(0.5) * torch.norm(vec, p=2, dim=-2).pow(2)
        return advantage.squeeze(-1)


class TrilMatrixModule(nn.Module):
    """Neural network module which outputs a lower-triangular matrix."""

    def __init__(self, logit_dim, matrix_dim):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.row_sizes = list(range(1, self.matrix_dim + 1))
        tril_dim = int(self.matrix_dim * (self.matrix_dim + 1) / 2)
        self.linear_module = nn.Linear(logit_dim, tril_dim)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        flat_tril = self.linear_module(logits)
        # Split flat lower triangular into rows
        split_tril = torch.split(flat_tril, self.row_sizes, dim=-1)
        # Compute exponentiated diagonals, row by row
        exp_tril = []
        for row in split_tril:
            exp_tril.append(torch.cat([row[..., :-1], row[..., -1:].exp()], dim=-1))
        # Fill upper triangular with zeros, row by row
        pad_tril = []
        for row in exp_tril:
            zeros = torch.zeros(row.shape[:-1] + (self.matrix_dim - row.shape[-1],))
            pad_tril.append(torch.cat([row, zeros], dim=-1))
        # Stack rows into a single (batched) matrix. dim=-2 ensures that we stack then
        # as rows, not columns (which would effectively transpose the matrix into an
        # upper triangular one)
        tril = torch.stack(pad_tril, dim=-2)
        return tril


class StateActionEncodingModule(nn.Module):
    """Neural network module which concatenates action after the first layer."""

    def __init__(self, obs_dim, action_dim, units=(), activation="relu"):
        super().__init__()
        if units:
            self.obs_module = nn.Sequential(nn.Linear(obs_dim, units[0]), activation())
            input_dim = units[0] + action_dim
            units = units[1:]
            self.sequential_module = FullyConnectedModule(
                input_dim, units=units, activation=activation
            )
        else:
            self.obs_module = nn.Identity()
            self.sequential_module = nn.Identity()

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        output = self.obs_module(obs)
        output = torch.cat([output, actions], dim=-1)
        output = self.sequential_module(output)
        return output


class FullyConnectedModule(nn.Module):
    """Neural network module that applies several fully connected modules to inputs."""

    def __init__(self, input_dim, units=(), activation="relu"):
        super().__init__()
        activation = get_activation(activation)
        units = [input_dim] + units
        modules = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation())
        self.sequential_module = nn.Sequential(*modules)

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
