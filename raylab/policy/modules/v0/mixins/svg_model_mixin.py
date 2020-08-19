"""SVG model architecture."""
from dataclasses import dataclass

import torch
import torch.nn as nn
from gym.spaces import Box
from ray.rllib.utils import override

import raylab.torch.nn as nnx
import raylab.torch.nn.distributions as ptd
from raylab.policy.modules.model.stochastic.single import ResidualMixin
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.policy.modules.networks.mlp import StateActionMLP
from raylab.torch.nn.init import initialize_


@dataclass
class SVGModelSpec(StateActionMLP.spec_cls):
    """Specifications for stochastic mlp model network.

    Inherits parameters from `StateActionMLP.spec_cls`.

    Args:
        units: Number of units in each hidden layer
        activation: Nonlinearity following each linear layer
        delay_action: Whether to apply an initial preprocessing layer on the
            observation before concatenating the action to the input.
        standard_scaler: Whether to transform the inputs of the NN using a
            standard scaling procedure (subtract mean and divide by stddev). The
            transformation mean and stddev should be fitted during training and
            used for both training and evaluation.
        input_dependent_scale: Whether to parameterize the Gaussian standard
            deviation as a function of the state and action
        residual: Whether to build model as a residual one, i.e., that
            predicts the change in state rather than the next state itself
    """

    input_dependent_scale: bool = True
    residual: bool = True


class SVGModelMixin:
    # pylint:disable=missing-docstring,too-few-public-methods
    @staticmethod
    def _make_model(obs_space, action_space, config):
        spec = SVGModelSpec.from_dict(config.get("model", {}))

        if spec.residual:
            model = ResidualSVGModel(obs_space, action_space, spec)
        else:
            model = SVGModel(obs_space, action_space, spec)

        return {"model": model}


class SVGModel(StochasticModel):
    """Model from Stochastic Value Gradients."""

    spec_cls = SVGModelSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: SVGModelSpec):
        params = SVGDynamicsParams(obs_space, action_space, spec)
        dist = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)
        super().__init__(params, dist)

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all encoder parameters.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        self.params.initialize_parameters(initializer_spec)


class ResidualSVGModel(ResidualMixin, SVGModel):
    # pylint:disable=missing-docstring
    pass


class SVGDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to distribution parameters through parallel
    subnetworks for each output dimension.
    """

    spec_cls = SVGModelSpec

    def __init__(self, obs_space, action_space, spec: SVGModelSpec):
        super().__init__()

        def make_encoder():
            return StateActionMLP(obs_space, action_space, spec)

        self.logits = nn.ModuleList([make_encoder() for _ in range(obs_space.shape[0])])
        self._activation = spec.activation

        def make_param(in_features):
            kwargs = dict(
                event_size=1, input_dependent_scale=spec.input_dependent_scale
            )
            return nnx.NormalParams(in_features, **kwargs)

        self.params = nn.ModuleList([make_param(m.out_features) for m in self.logits])

    @override(nn.Module)
    def forward(self, obs, act):  # pylint:disable=arguments-differ
        params = [p(l(obs, act)) for p, l in zip(self.params, self.logits)]
        loc = torch.cat([d["loc"] for d in params], dim=-1)
        scale = torch.cat([d["scale"] for d in params], dim=-1)
        return {"loc": loc, "scale": scale}

    def initialize_parameters(self, initializer_spec: dict):
        # pylint:disable=missing-docstring
        self.logits.apply(initialize_(activation=self._activation, **initializer_spec))
