"""Parameterized deterministic policies."""
import warnings
from typing import Optional

import torch
import torch.nn as nn
from gym.spaces import Box
from torch import Tensor

import raylab.pytorch.nn as nnx
from raylab.modules.networks.mlp import StateMLP


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
        if self.noise is not None:
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

    spec_cls = StateMLP.spec_cls

    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        mlp_spec: StateMLP.spec_cls,
        norm_beta: float,
    ):
        encoder = StateMLP(obs_space, mlp_spec)

        action_size = action_space.shape[0]
        if norm_beta:
            action_linear = nnx.NormalizedLinear(
                encoder.out_features, action_size, beta=norm_beta
            )
        else:
            action_linear = nn.Linear(encoder.out_features, action_size)

        action_low, action_high = map(
            torch.as_tensor, (action_space.low, action_space.high)
        )
        squash = nnx.TanhSquash(action_low, action_high)

        super().__init__(encoder, action_linear, squash)

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Uses `raylab.pytorch.nn.init.initialize_` to create an initializer
        function.

        Args:
            initializer_spec: Dictionary with mandatory `type` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        self.encoder.initialize_parameters(initializer_spec)
