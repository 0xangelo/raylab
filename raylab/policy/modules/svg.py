"""Architecture used in Stochastic Value Gradients."""
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .actor import StochasticActor
from .critic import MLPVValue
from .model import ResidualSVGModel
from .model import SVGModel


ModelSpec = SVGModel.spec_cls
ActorSpec = StochasticActor.spec_cls
CriticSpec = MLPVValue.spec_cls


@dataclass
class SVGSpec(DataClassJsonMixin):
    # pylint:disable=missing-class-docstring
    model: ModelSpec = field(default_factory=ModelSpec)
    actor: ActorSpec = field(default_factory=ActorSpec)
    critic: CriticSpec = field(default_factory=CriticSpec)


class SVG(nn.Module):
    """Architecture used in Stochastic Value Gradients."""

    # pylint:disable=abstract-method
    spec_cls = SVGSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: SVGSpec):
        super().__init__()
        self._make_model(obs_space, action_space, spec.model)
        self._make_actor(obs_space, action_space, spec.actor)
        self._make_critic(obs_space, spec.critic)

    def _make_model(self, obs_space: Box, action_space: Box, spec: ModelSpec):
        if spec.residual:
            self.model = ResidualSVGModel(obs_space, action_space, spec)
        else:
            self.model = SVGModel(obs_space, action_space, spec)

    def _make_actor(self, obs_space: Box, action_space: Box, spec: ActorSpec):
        actor = StochasticActor(obs_space, action_space, spec)
        self.actor = actor.policy

    def _make_critic(self, obs_space: Box, spec: CriticSpec):
        critic = MLPVValue(obs_space, spec)
        target_critic = MLPVValue(obs_space, spec)
        target_critic.load_state_dict(critic.state_dict())

        self.critic = critic
        self.target_critic = target_critic


class SoftSVG(SVG):
    """Architecture used for SVG in the Maximum Entropy framework."""

    def _make_actor(self, obs_space: Box, action_space: Box, spec: ActorSpec):
        actor = StochasticActor(obs_space, action_space, spec)
        self.actor = actor.policy
        self.alpha = actor.alpha
