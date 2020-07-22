"""NN architecture used in Normalized Advantage Function."""
from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box

from .actor.policy.deterministic import MLPDeterministicPolicy
from .critic.naf_value import NAFQValue
from .critic.q_value import ForkedQValueEnsemble
from .critic.q_value import QValueEnsemble
from .critic.v_value import ForkedVValueEnsemble
from .critic.v_value import VValueEnsemble


MLPPolicySpec = MLPDeterministicPolicy.spec_cls


@dataclass
class NAFSpec(DataClassJsonMixin):
    """Specifications for Normalized Advantage Function."""

    policy: MLPPolicySpec = field(default_factory=MLPPolicySpec)
    separate_behavior: bool = False
    double_q: bool = True
    parallelized: bool = False


class NAF(nn.Module):
    """Neural network architecture for Normalized Advantage Function."""

    # pylint:disable=abstract-method
    spec_cls = NAFSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: NAFSpec):
        super().__init__()

        def make_policy():
            return MLPDeterministicPolicy(obs_space, action_space, spec.policy)

        policies = [make_policy() for _ in range(1 + spec.double_q)]
        self.actor = policies[0]
        self.behavior = self.actor
        if spec.separate_behavior:
            self.behavior = make_policy()
            self.behavior.load_state_dict(self.actor.state_dict())

        ensemble_cls = ForkedQValueEnsemble if spec.parallelized else QValueEnsemble
        self.critics = ensemble_cls([NAFQValue(action_space, pol) for pol in policies])

        ensemble_cls = ForkedVValueEnsemble if spec.parallelized else VValueEnsemble
        self.vcritics = ensemble_cls([q.v_value for q in self.critics])
        self.target_vcritics = ensemble_cls(
            [NAFQValue(action_space, make_policy()).v_value for _ in self.critics]
        )
        self.target_vcritics.load_state_dict(self.vcritics.state_dict())
