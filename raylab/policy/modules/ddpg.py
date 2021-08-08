"""NN architecture used in Deep Deterministic Policy Gradients."""
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from nnrl.nn.actor.deterministic import DeterministicActor
from nnrl.nn.critic import ActionValueCritic
from torch import nn

ActorSpec = DeterministicActor.spec_cls
CriticSpec = ActionValueCritic.spec_cls


@dataclass
class DDPGSpec(DataClassJsonMixin):
    """Specifications for DDPG modules.

    Args:
        actor: Specifications for policy, behavior, and target policy
        critic: Specifications for action-value estimators
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Overrides actor and critic initializer
            specifications.
    """

    actor: ActorSpec = field(default_factory=ActorSpec)
    critic: CriticSpec = field(default_factory=CriticSpec)
    initializer: dict = field(default_factory=dict)

    def __post_init__(self):
        # Top-level initializer options take precedence over individual
        # component's options
        if self.initializer:
            self.actor.initializer = self.initializer
            self.critic.initializer = self.initializer


class DDPG(nn.Module):
    """NN module for DDPG-like algorithms.

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
    spec_cls = DDPGSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: DDPGSpec):
        super().__init__()
        # Build actor
        actor = DeterministicActor(obs_space, action_space, spec.actor)
        self.actor = actor.policy
        self.behavior = actor.behavior
        self.target_actor = actor.target_policy

        # Build critic
        critic = ActionValueCritic(obs_space, action_space, spec.critic)
        self.critics = critic.q_values
        self.target_critics = critic.target_q_values
