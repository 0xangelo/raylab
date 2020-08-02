"""Constructors for stochastic dynamics models."""
from dataclasses import dataclass
from dataclasses import field

from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .ensemble import ForkedSME
from .ensemble import SME
from .single import MLPModel
from .single import ResidualMLPModel

ModelSpec = MLPModel.spec_cls


@dataclass
class Spec(DataClassJsonMixin):
    """Specifications for stochastic dynamics model.

    Args:
        network: Specifications for stochastic model network
        residual: Whether to build model as a residual one, i.e., that
            predicts the change in state rather than the next state itself
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Used to initialize the model's Linear layers.
    """

    network: ModelSpec = field(default_factory=ModelSpec)
    residual: bool = True
    initializer: dict = field(default_factory=dict)


def build(obs_space: Box, action_space: Box, spec: Spec) -> MLPModel:
    """Construct stochastic dynamics model.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for stochastic dynamics model

    Returns:
        A stochastic dynamics model
    """
    cls = ResidualMLPModel if spec.residual else MLPModel
    model = cls(obs_space, action_space, spec.network)
    model.initialize_parameters(spec.initializer)
    return model


@dataclass
class EnsembleSpec(Spec):
    """Specifications for stochastic dynamics model ensemble.

    Args:
        network: Specifications for stochastic model networks
        residual: Whether to build each model as a residual one, i.e., that
            predicts the change in state rather than the next state itself
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments. Used to initialize the models' Linear layers.
        ensemble_size: Number of models in the collection.
        parallelize: Whether to use an ensemble with parallelized `sample`,
            `rsample`, and `log_prob` methods
    """

    ensemble_size: int = 1
    parallelize: bool = False


def build_ensemble(obs_space: Box, action_space: Box, spec: EnsembleSpec) -> SME:
    """Construct stochastic dynamics model ensemble.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for stochastic dynamics model ensemble

    Returns:
        A stochastic dynamics model ensemble
    """
    models = [build(obs_space, action_space, spec) for _ in range(spec.ensemble_size)]
    cls = ForkedSME if spec.parallelize else SME
    ensemble = cls(models)
    return ensemble
