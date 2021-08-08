"""NN architecture used in Deep Deterministic Policy Gradients."""
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from nnrl.nn.actor.deterministic import DeterministicActor
from nnrl.nn.critic import ActionValueCritic
from torch import nn


@dataclass
class ActorSpec(DeterministicActor.spec_cls):
    """Specifications for policy, behavior, and target policy."""

    separate_target_policy: bool = field(default=False, init=False, repr=False)
    smooth_target_policy: bool = field(default=True, init=False, repr=False)


CriticSpec = ActionValueCritic.spec_cls


def default_actor() -> ActorSpec:
    spec = ActorSpec()
    spec.network.units = (256, 256)
    spec.network.activation = "ReLU"
    spec.network.layer_norm = False
    spec.network.norm_beta = 1.2
    spec.separate_behavior = False
    spec.target_gaussian_sigma = 0.3
    spec.initializer = {}
    return spec


def default_critic() -> CriticSpec:
    spec = CriticSpec()
    spec.encoder.units = (256, 256)
    spec.encoder.activation = "ReLU"
    spec.encoder.delay_action = False
    spec.double_q = True
    spec.parallelize = True
    spec.initializer = {}
    return spec


@dataclass
class SOPSpec(DataClassJsonMixin):
    """Specifications for SOP modules.

    Args:
        actor: Specifications for policy, behavior, and target policy
        critic: Specifications for action-value estimators
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Overrides actor and critic initializer
            specifications.
    """

    actor: ActorSpec = field(default_factory=default_actor)
    critic: CriticSpec = field(default_factory=default_critic)
    initializer: dict = field(default_factory=dict)

    def __post_init__(self):
        # Top-level initializer options take precedence over individual
        # component's options
        if self.initializer:
            self.actor.initializer = self.initializer
            self.critic.initializer = self.initializer


class SOP(nn.Module):
    """NN module for SOP-like algorithms.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for DDPG modules

    Attributes:
        actor (DeterministicPolicy): The deterministic policy to be learned
        behavior (DeterministicPolicy): The policy for exploration
        target_actor (DeterministicPolicy): The policy used for estimating the
            arg max in Q-Learning
        critics (QValueEnsemble): The action-value estimators to be learned
        target_critics (QValueEnsemble): The action-value estimators used for
            bootstrapping in Q-Learning
        spec_cls: Expected class of `spec` init argument
    """

    # pylint:disable=abstract-method
    spec_cls = SOPSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: SOPSpec):
        super().__init__()
        # Build actor
        actor = DeterministicActor(obs_space, action_space, spec.actor)
        self.actor = actor.policy
        self.behavior = actor.behavior
        self.target_actor = actor.target_policy

        main, target = set(self.actor.parameters()), set(self.target_actor.parameters())
        symdiff = main.symmetric_difference(target)
        assert not symdiff, f"Main and target policy must be the same: {symdiff}"

        # Build critic
        critic = ActionValueCritic(obs_space, action_space, spec.critic)
        self.critics = critic.q_values
        self.target_critics = critic.target_q_values
