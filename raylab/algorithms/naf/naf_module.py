"""Normalized Advantage Function nn.Module."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class NAFModule(nn.Module):
    """Neural network module that implements the forward pass of NAF."""

    def __init__(self, obs_dim, action_low, action_high, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_low.numel()

        self.logits_module = FullyConnectedModule(
            self.obs_dim, units=config["layers"], activation=config["activation"]
        )
        logit_dim = self.logits_module.output_dim
        self.value_module = ValueModule(logit_dim)
        self.action_module = ActionModule(logit_dim, action_low, action_high)
        self.advantage_module = AdvantageModule(logit_dim, self.action_dim)

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs, actions)
        best_value = self.value_module(logits)
        best_action = self.action_module(logits)
        advantage = self.advantage_module(logits, best_action, actions)
        action_value = best_value + advantage
        return action_value, best_action, best_value


class ValueModule(nn.Module):
    """Neural network module implementing the value function term of NAF."""

    def __init__(self, logit_dim):
        super().__init__()
        self.logit_dim = logit_dim
        self.linear_module = nn.Linear(self.logit_dim, 1)
        self.output_dim = 1

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        return self.linear_module(logits)


class ActionModule(nn.Module):
    """Neural network module implementing the greedy action term of NAF."""

    def __init__(self, logit_dim, action_low, action_high):
        super().__init__()
        self.logit_dim = logit_dim
        self.action_low = action_low
        self.action_range = action_high - action_low
        action_dim = self.action_low.numel()
        self.linear_module = nn.Linear(self.logit_dim, action_dim)

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
        self.logit_dim = logit_dim
        self.action_dim = action_dim
        self.tril_matrix_module = TrilMatrixModule(self.logit_dim, self.action_dim)

    @override(nn.Module)
    def forward(self, logits, best_action, actions):  # pylint: disable=arguments-differ
        tril_matrix = self.tril_matrix_module(logits)
        pdef_matrix = torch.matmul(tril_matrix, tril_matrix.transpose(-1, -2))
        action_diff = actions - best_action
        action_diff.unsqueeze_(-1)  # column vector
        quadratic_term = torch.matmul(
            action_diff.transpose(-1, -2), torch.matmul(pdef_matrix, action_diff)
        )
        advantage = -1 / 2 * quadratic_term
        return advantage.squeeze()


class TrilMatrixModule(nn.Module):
    """Neural network module which outputs a lower-triangular matrix."""

    def __init__(self, logit_dim, matrix_dim):
        super().__init__()
        self.logit_dim = logit_dim
        self.matrix_dim = matrix_dim
        tril_dim = int(self.matrix_dim * (self.matrix_dim + 1) / 2)
        self.linear_module = nn.Linear(self.logit_dim, tril_dim)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        flat_triangular = self.linear_module(logits)
        flat_triangular.unsqueeze_(-2)
        matrix_shape = logits.shape[:-1] + (self.matrix_dim, self.matrix_dim)
        tril_indices = torch.tril_indices(self.matrix_dim, self.matrix_dim).split(1)
        tril_indices = (...,) + tril_indices
        tril_matrix = torch.zeros(*matrix_shape)
        tril_matrix[tril_indices] = flat_triangular
        diag_indices = (torch.arange(self.matrix_dim),) * 2
        diag_indices = (...,) + diag_indices
        tril_matrix[diag_indices] = tril_matrix[diag_indices].exp()
        return tril_matrix


class StateActionEncodingModule(nn.Module):
    """Neural network module which concatenates action after the first layer."""

    def __init__(self, obs_dim, action_dim, units=(), activation="relu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if units:
            self.obs_module = nn.Sequential(
                nn.Linear(self.obs_dim, units[0]), activation()
            )
            input_dim = units[0] + self.action_dim
            units = units[1:]
            self.sequential_module = FullyConnectedModule(
                input_dim, units=units, activation=activation
            )
            self.output_dim = self.sequential_module.output_dim
        else:
            self.obs_module = nn.Identity()
            self.sequential_module = nn.Identity()
            self.output_dim = self.obs_dim + self.action_dim

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
        self.input_dim = input_dim

        activation = get_activation(activation)
        units = [self.input_dim] + units
        modules = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation())
        self.sequential_module = nn.Sequential(*modules)
        self.output_dim = units[-1]

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
