"""Actor-Critic architecture popularized by DDPG."""
from .actor_critic import AbstractActorCritic
from .deterministic_actor_mixin import DeterministicActorMixin
from .action_value_mixin import ActionValueMixin


# pylint:disable=abstract-method
class DDPGModule(DeterministicActorMixin, ActionValueMixin, AbstractActorCritic):
    """Actor-Critic module with deterministic actor and action-value critics."""


# pylint:enable=abstract-method
