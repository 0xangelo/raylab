"""Network and configurations for model-based TD3 algorithms."""
from dataclasses import dataclass, field

from gym.spaces import Box
from nnrl.nn.model import EnsembleSpec, build_ensemble

from .sop import SOP, ActorSpec, CriticSpec, SOPSpec


def default_models() -> EnsembleSpec:
    """Model configurations used in MAGE paper for Mujoco environments."""
    spec = EnsembleSpec()
    spec.network.units = (512,) * 4
    spec.network.activation = "Swish"
    spec.network.delay_action = False
    spec.network.fix_logvar_bounds = True
    spec.network.input_dependent_scale = True
    spec.residual = True
    spec.ensemble_size = 8
    spec.parallelize = True
    return spec


def default_actor() -> ActorSpec:
    """Actor configurations used in MAGE paper for Mujoco environments."""
    spec = ActorSpec()
    spec.network.units = 2 * (284,)
    spec.network.activation = "ReLU"
    spec.network.layer_norm = False
    spec.network.norm_beta = 1.2
    spec.separate_behavior = False
    spec.target_gaussian_sigma = 0.3
    return spec


def default_critic() -> CriticSpec:
    """Critic configurations used in MAGE paper for Mujoco environments."""
    spec = CriticSpec()
    spec.encoder.units = 2 * (384,)
    spec.encoder.activation = "ReLU"
    spec.encoder.delay_action = False
    spec.double_q = True
    spec.parallelize = True
    return spec


@dataclass
class MAGESpec(SOPSpec):
    """Specifications for NNs used in MAGE.

    Args:
        model: Specifications for stochastic dynamics model ensemble
        actor: Specifications for policy, behavior, and target policy
        critic: Specifications for action-value estimators
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Overrides model, actor, and critic initializer
            specifications.
    """

    model: EnsembleSpec = field(default_factory=default_models)
    actor: ActorSpec = field(default_factory=default_actor)
    critic: CriticSpec = field(default_factory=default_critic)

    def __post_init__(self):
        super().__post_init__()
        if self.initializer:
            self.model.initializer = self.initializer


class MAGE(SOP):
    """NN module for MAGE algorithm

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for NNs used in MAGE

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
    spec_cls = MAGESpec

    def __init__(self, obs_space: Box, action_space: Box, spec: MAGESpec):
        super().__init__(obs_space, action_space, spec)

        self.models = build_ensemble(obs_space, action_space, spec.model)
