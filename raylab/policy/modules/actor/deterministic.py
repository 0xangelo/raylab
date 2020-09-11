"""Network and configurations for modules with deterministic policies."""
import warnings
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .policy.deterministic import DeterministicPolicy
from .policy.deterministic import MLPDeterministicPolicy

NetworkSpec = MLPDeterministicPolicy.spec_cls


@dataclass
class DeterministicActorSpec(DataClassJsonMixin):
    """Specifications for policy, behavior, and target policy.

    Args:
        network: Specifications for deterministic mlp policy network
        separate_behavior: Whether to create a separate behavior policy. Usually
            for parameter noise exploration, in which case it is recommended to
            enable encoder.layer_norm alongside this option.
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

    network: NetworkSpec = field(default_factory=NetworkSpec)
    separate_behavior: bool = False
    smooth_target_policy: bool = True
    target_gaussian_sigma: float = 0.3
    separate_target_policy: bool = False
    initializer: dict = field(default_factory=dict)

    def __post_init__(self):
        cls_name = type(self).__name__
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
        behavior: The policy for exploration. `utils.exploration.GaussianNoise`
            handles Gaussian action noise exploration separatedly.
        target_policy: The policy used for estimating the arg max in Q-Learning
        spec_cls: Expected class of `spec` init argument
    """

    # pylint:disable=abstract-method
    spec_cls = DeterministicActorSpec

    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        spec: DeterministicActorSpec,
    ):
        super().__init__()

        def make_policy():
            return MLPDeterministicPolicy(obs_space, action_space, spec.network)

        policy = make_policy()
        policy.initialize_parameters(spec.initializer)

        behavior = policy
        if spec.separate_behavior:
            if not spec.network.layer_norm:
                warnings.warn(
                    "Separate behavior policy requested and layer normalization"
                    " deactivated. If using parameter noise exploration, enable"
                    " layer normalization for better stability."
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
