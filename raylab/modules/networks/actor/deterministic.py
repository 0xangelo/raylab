"""Network and configurations for modules with deterministic policies."""
import warnings
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .policy.deterministic import DeterministicPolicy
from .policy.deterministic import MLPDeterministicPolicy

MLPSpec = MLPDeterministicPolicy.spec_cls


@dataclass
class DeterministicActorSpec(DataClassJsonMixin):
    """Specifications for policy, behavior, and target policy.

    Args:
        encoder: Specifications for creating the multilayer perceptron mapping
            states to pre-action linear features
        norm_beta: Maximum l1 norm of the unconstrained actions. If None, won't
            normalize actions before squashing function
        behavior: Type of behavior policy. Either 'gaussian', 'parameter_noise',
            or 'deterministic'
        smooth_target_policy: Whether to use a noisy target policy for
            Q-Learning
        target_gaussian_sigma: Gaussian standard deviation for noisy target
            policy
        separate_target_policy: Whether to use separate parameters for the
            target policy. Intended for use with polyak averaging
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments.
    """

    encoder: MLPSpec = field(default_factory=MLPSpec)
    norm_beta: float = 1.2
    behavior: str = "gaussian"
    smooth_target_policy: bool = True
    target_gaussian_sigma: float = 0.3
    separate_target_policy: bool = False
    initializer: dict = field(default_factory=dict)

    def __post_init__(self):
        cls_name = type(self).__name__
        assert self.norm_beta > 0, f"{cls_name}.norm_beta must be positive"
        valid_behaviors = {"gaussian", "parameter_noise", "deterministic"}
        assert (
            self.behavior in valid_behaviors
        ), f"{cls_name}.behavior must be one of {valid_behaviors}"
        assert (
            self.target_gaussian_sigma > 0
        ), f"{cls_name}.target_gaussian_sigma must be positive"


class DeterministicActor(nn.Module):
    """NN with deterministic policies.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for policy, behavior, and target policy

    Attributes:
        policy: The deterministic policy to be learned
        behavior: The policy for exploration
        target_policy: The policy used for estimating the arg max in Q-Learning
        spec_cls: Expected class of `spec` init argument
    """

    # pylint:disable=abstract-method
    spec_cls = DeterministicActorSpec

    def __init__(
        self, obs_space: Box, action_space: Box, spec: DeterministicActorSpec,
    ):
        super().__init__()

        def make_policy():
            return MLPDeterministicPolicy(
                obs_space, action_space, spec.encoder, spec.norm_beta
            )

        policy = make_policy()
        policy.initialize_parameters(spec.initializer)

        behavior = policy
        if spec.behavior == "parameter_noise":
            if not spec.encoder.layer_norm:
                warnings.warn(
                    f"Behavior is set to {spec.behavior} but layer normalization is "
                    "deactivated. Use layer normalization for better stability."
                )
            behavior = make_policy()
            behavior.load_state_dict(policy.state_dict())

        target_policy = policy
        if spec.separate_target_policy:
            target_policy = make_policy()
            target_policy.load_state_dict(policy.state_dict())
        if spec.smooth_target_policy:
            target_policy = DeterministicPolicy.add_gaussian_noise(
                target_policy, noise_stddev=spec.target_gaussian_sigma
            )

        self.policy = policy
        self.behavior = behavior
        self.target_policy = target_policy