"""Network and configurations for modules with Q-value critics."""
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .q_value import ForkedQValueEnsemble
from .q_value import MLPQValue
from .q_value import QValueEnsemble


QValueSpec = MLPQValue.spec_cls


@dataclass
class ActionValueCriticSpec(DataClassJsonMixin):
    """Specifications for action-value estimators.

    Args:
        encoder: Specifications for creating the multilayer perceptron mapping
            states and actions to pre-value function linear features
        double_q: Whether to create two Q-value estimators instead of one.
            Defaults to True
        parallelize: Whether to evaluate Q-values in parallel. Defaults to
            False.
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments.
    """

    encoder: QValueSpec = field(default_factory=QValueSpec)
    double_q: bool = True
    parallelize: bool = False
    initializer: dict = field(default_factory=dict)


class ActionValueCritic(nn.Module):
    """NN with Q-value estimators.

    Since it is common to use clipped double Q-Learning, `q_values` is a
    ModuleList of Q-value functions.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for action-value estimators

    Attributes:
        q_values: The action-value estimators to be learned
        target_q_values: The action-value estimators used for bootstrapping in
            Q-Learning
        spec_cls: Expected class of `spec` init argument
    """

    # pylint:disable=abstract-method
    spec_cls = ActionValueCriticSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: ActionValueCriticSpec):
        super().__init__()

        def make_q_value():
            return MLPQValue(obs_space, action_space, spec.encoder)

        def make_q_value_ensemble():
            n_q_values = 2 if spec.double_q else 1
            q_values = [make_q_value() for _ in range(n_q_values)]

            if spec.parallelize:
                return ForkedQValueEnsemble(q_values)
            return QValueEnsemble(q_values)

        q_values = make_q_value_ensemble()
        q_values.initialize_parameters(spec.initializer)

        target_q_values = make_q_value_ensemble()
        main, target = set(q_values.parameters()), set(target_q_values.parameters())
        assert not main.intersection(
            target
        ), "Main and target Q nets cannot share params."
        target_q_values.load_state_dict(q_values.state_dict())
        for par in target:
            par.requires_grad = False

        self.q_values = q_values
        self.target_q_values = target_q_values
