"""Network and configurations for model-based DDPG algorithms."""
from dataclasses import dataclass, field

from gym.spaces import Box
from nnrl.nn.model import EnsembleSpec, build_ensemble

from .ddpg import DDPG, DDPGSpec


@dataclass
class ModelBasedDDPGSpec(DDPGSpec):
    """Specifications for model-based DDPG modules.

    Args:
        model: Specifications for stochastic dynamics model ensemble
        actor: Specifications for policy, behavior, and target policy
        critic: Specifications for action-value estimators
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Overrides model, actor, and critic initializer
            specifications.
    """

    model: EnsembleSpec = field(default_factory=EnsembleSpec)

    def __post_init__(self):
        super().__post_init__()
        if self.initializer:
            self.model.initializer = self.initializer


class ModelBasedDDPG(DDPG):
    """NN module for Model-Based DDPG algorithms.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for model-based DDPG modules

    Attributes:
        model (SME): Stochastic dynamics model ensemble
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
    spec_cls = ModelBasedDDPGSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: ModelBasedDDPGSpec):
        super().__init__(obs_space, action_space, spec)

        self.models = build_ensemble(obs_space, action_space, spec.model)
