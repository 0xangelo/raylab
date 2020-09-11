"""Architecture used in Trust Region Policy Optimization."""
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .actor import StochasticActor
from .critic import MLPVValue


ActorSpec = StochasticActor.spec_cls
CriticSpec = MLPVValue.spec_cls


@dataclass
class TRPOSpec(DataClassJsonMixin):
    # pylint:disable=missing-class-docstring
    actor: ActorSpec = field(default_factory=ActorSpec)
    critic: CriticSpec = field(default_factory=CriticSpec)


class TRPO(nn.Module):
    # pylint:disable=missing-class-docstring
    spec_cls = TRPOSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: TRPOSpec):
        super().__init__()
        self._make_actor(obs_space, action_space, spec.actor)
        self._make_critic(obs_space, spec.critic)

    def _make_actor(self, obs_space: Box, action_space: Box, spec: ActorSpec):
        actor = StochasticActor(obs_space, action_space, spec)
        self.actor = actor.policy

    def _make_critic(self, obs_space: Box, spec: CriticSpec):
        self.critic = MLPVValue(obs_space, spec)
