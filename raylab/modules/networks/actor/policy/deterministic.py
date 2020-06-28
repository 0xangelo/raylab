"""Parameterized deterministic policies."""
import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional

import torch
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor

import raylab.pytorch.nn as nnx
from raylab.pytorch.nn.init import initialize_


class DeterministicPolicy(nn.Module):
    """Continuous action deterministic policy as a sequence of modules.

    If a noise module is passed, it is evaluated on unconstrained actions before
    the squashing module.

    Args:
        encoder: NN module mapping states to 1D features
        action_linear: Linear module mapping features to unconstrained actions
        squashing: Invertible module mapping unconstrained actions to bounded
            action space
        noise: Optional stochastic module adding noise to unconstrained actions
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_linear: nn.Module,
        squashing: nn.Module,
        noise: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_linear = action_linear
        self.squashing = squashing
        self.noise = noise

    def forward(self, obs: Tensor) -> Tensor:  # pylint:disable=arguments-differ
        """Main forward pass mapping observations to actions."""
        unconstrained_action = self.unconstrained_action(obs)
        return self.squashing(unconstrained_action)

    @torch.jit.export
    def unconstrained_action(self, obs: Tensor) -> Tensor:
        """Forward pass with no squashing at the end"""
        features = self.encoder(obs)
        unconstrained_action = self.action_linear(features)
        if self.noise:
            unconstrained_action = self.noise(unconstrained_action)
        return unconstrained_action

    @torch.jit.export
    def unsquash_action(self, action: Tensor) -> Tensor:
        """Returns the unconstrained action which generated the given action."""
        return self.squashing(action, reverse=True)

    @classmethod
    def add_gaussian_noise(cls, policy, noise_stddev: float):
        """Adds a zero-mean Gaussian noise module to a DeterministicPolicy.

        Args:
            policy: The deterministic policy.
            noise_stddev: Standard deviation of the Gaussian noise

        Returns:
            A deterministic policy sharing all paremeters with the input one and
            with additional noise module before squashing.
        """
        if policy.noise is not None:
            warnings.warn(
                "Adding Gaussian noise to already noisy policy. Are you sure you"
                " called `add_gaussian_noise` on the right policy?"
            )
        noise = nnx.GaussianNoise(noise_stddev)
        return cls(policy.encoder, policy.action_linear, policy.squashing, noise=noise)


@dataclass
class StateMLPSpec:
    """Specifications for creating a multilayer perceptron.

    Args:
    units: Number of units in each hidden layer
    activation: Nonlinearity following each linear layer
    layer_norm: Whether to apply layer normalization between each linear layer
        and following activation
    """

    units: List[int]
    activation: Optional[str]
    layer_norm: bool


class MLPDeterministicPolicy(DeterministicPolicy):
    """DeterministicPolicy with multilayer perceptron encoder.

    The final Linear layer is initialized so that actions are near the origin
    point.

    Args:
        obs_space: Observation space
        action_space: Action space
        mlp_spec: Multilayer perceptron specifications
        norm_beta: Maximum l1 norm of the unconstrained actions. If None, won't
            normalize actions before squashing function.

    Attributes:
        spec_cls: Expected class of `spec` init argument
    """

    spec_cls = StateMLPSpec

    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        mlp_spec: StateMLPSpec,
        norm_beta: float,
    ):
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]
        action_low, action_high = map(
            torch.as_tensor, (action_space.low, action_space.high)
        )

        encoder = nnx.FullyConnected(
            obs_size,
            mlp_spec.units,
            mlp_spec.activation,
            layer_norm=mlp_spec.layer_norm,
        )

        if norm_beta:
            action_linear = nnx.NormalizedLinear(
                encoder.out_features, action_size, beta=norm_beta
            )
        else:
            action_linear = nn.Linear(encoder.out_features, action_size)

        squash = nnx.TanhSquash(action_low, action_high)

        super().__init__(encoder, action_linear, squash)
        self.mlp_spec = mlp_spec

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Uses `raylab.pytorch.nn.init.initialize_` to create an initializer
        function.

        Args:
            initializer_spec: Dictionary with mandatory `type` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        initializer = initialize_(
            activation=self.mlp_spec.activation, **initializer_spec
        )
        self.encoder.apply(initializer)
