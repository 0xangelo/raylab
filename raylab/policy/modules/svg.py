"""Architecture used in Stochastic Value Gradients."""
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from nnrl.nn.actor import StochasticActor
from nnrl.nn.critic import MLPVValue
from nnrl.nn.model import ResidualStochasticModel, SVGModel
from torch import nn

ModelSpec = SVGModel.spec_cls
CriticSpec = MLPVValue.spec_cls


@dataclass
class ActorSpec(StochasticActor.spec_cls):
    # pylint:disable=missing-class-docstring
    old_policy: bool = False


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
        model = SVGModel(obs_space, action_space, spec)
        self.model = ResidualStochasticModel(model) if spec.residual else model

    def _make_actor(self, obs_space: Box, action_space: Box, spec: ActorSpec):
        actor = StochasticActor(obs_space, action_space, spec)
        self.actor = actor.policy
        if spec.old_policy:
            self.old_actor = StochasticActor(obs_space, action_space, spec).policy

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
