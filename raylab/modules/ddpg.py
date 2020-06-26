"""NN architecture used in Deep Deterministic Policy Gradients."""
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.utils.annotations import TensorDict

from .networks.action_value import ForkedQValueEnsemble
from .networks.action_value import MLPQValue
from .networks.action_value import QValueEnsemble
from .networks.action_value import StateActionMLPSpec
from .networks.policy.deterministic import DeterministicPolicy
from .networks.policy.deterministic import MLPDeterministicPolicy
from .networks.policy.deterministic import StateMLPSpec


@dataclass
class DDPGActorSpec(DataClassJsonMixin):
    """Specifications for policy, behavior, and target policy.

    Args:
        encoder: Specifications for creating the multilayer perceptron mapping
            states to pre-action linear features
        norm_beta: Maximum l1 norm of the unconstrained actions. If None, won't
            normalize actions before squashing function
        behavior: Type of behavior policy. Either 'gaussian', 'parameter_noise',
            or 'deterministic'
        smooth_target_policy: Whether to use a noisy target policy for
            Q-Learning
        target_gaussian_sigma: Gaussian standard deviation for noisy target
            policy
        separate_target_policy: Whether to use separate parameters for the
            target policy. Intended for use with polyak averaging
    """

    encoder: StateMLPSpec = field(default_factory=StateMLPSpec)
    norm_beta: float = 1.2
    behavior: str = "gaussian"
    smooth_target_policy: bool = True
    target_gaussian_sigma: float = 0.3
    separate_target_policy: bool = False

    def __post_init__(self):
        cls_name = type(self).__name__
        assert self.norm_beta > 0, f"{cls_name}.norm_beta must be positive"
        valid_behaviors = {"gaussian", "parameter_noise", "deterministic"}
        assert (
            self.behavior in valid_behaviors
        ), f"{cls_name}.behavior must be one of {valid_behaviors}"
        assert (
            self.target_gaussian_sigma > 0
        ), f"{cls_name}.target_gaussian_sigma must be positive"


@dataclass
class DDPGCriticSpec(DataClassJsonMixin):
    """Specifications for action-value estimators.

    Args:
        encoder: Specifications for creating the multilayer perceptron mapping
            states and actions to pre-value function linear features
        double_q: Whether to create two Q-value estimators instead of one.
            Defaults to True
        parallelize: Whether to evaluate Q-values in parallel. Defaults to
            False.
    """

    encoder: StateActionMLPSpec = field(default_factory=StateActionMLPSpec)
    double_q: bool = True
    parallelize: bool = False


@dataclass
class DDPGSpec(DataClassJsonMixin):
    """Specifications for DDPG modules.

    Args:
        actor: Specifications for policy, behavior, and target policy
        critic: Specifications for action-value estimators
        initializer: Optional dictionary with mandatory `type` key corresponding
            to the initializer function name in `torch.nn.init` and optional
            keyword arguments.configuration dictionary for parameter
    """

    actor: DDPGActorSpec = field(default_factory=DDPGActorSpec)
    critic: DDPGCriticSpec = field(default_factory=DDPGCriticSpec)
    initializer: dict = field(default_factory=dict)


class DDPG(nn.Module):
    """NN module for DDPG-like algorithms.

    Since it is common to use clipped double Q-Learning, critic is implemented as
    a ModuleList of action-value functions.

    Uses `raylab.pytorch.nn.init.initialize_` to create an initializer
    function for the parameters.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for DDPG modules

    Attributes:
        actor: The deterministic policy to be learned
        behavior: The policy for exploration
        target_actor: The policy used for estimating the arg max in Q-Learning
        critics: The action-value estimators to be learned
        target_critics: The action-value estimators used for bootstrapping in
            Q-Learning
        forward_batch_keys: Keys in the input tensor dict that will be accessed
            in the main forward pass. Useful for the caller to convert the
            necessary inputs to tensors
    """

    actor: DeterministicPolicy
    behavior: DeterministicPolicy
    target_actor: DeterministicPolicy
    critics: nn.ModuleList
    target_critics: nn.ModuleList
    forward_batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(self, obs_space: Box, action_space: Box, spec: DDPGSpec):
        super().__init__()
        # Build actor
        self.actor, self.behavior, self.target_actor = self._make_actor(
            obs_space, action_space, spec.actor, spec.initializer
        )

        # Build critic
        self.critics, self.target_critics = self._make_critic(
            obs_space, action_space, spec.critic, spec.initializer
        )

    def forward(
        self, input_dict: TensorDict, state: List[Tensor], seq_lens: Tensor
    ) -> Tuple[TensorDict, List[Tensor]]:
        """Maps input tensors to action distribution parameters.

        Args:
            input_dict: Tensor dictionary with mandatory `forward_batch_keys`
                contained within
            state: List of RNN state tensors
            seq_lens: 1D tensor holding input sequence lengths

        Returns:
            A tuple containg an input dictionary to the policy's `dist_class`
            and a list of RNN state tensors
        """
        # pylint:disable=unused-argument,arguments-differ
        return {"obs": input_dict["obs"]}, state

    @staticmethod
    def _make_actor(
        obs_space: Box, action_space: Box, spec: DDPGActorSpec, initializer_spec: dict
    ) -> Tuple[MLPDeterministicPolicy, MLPDeterministicPolicy, MLPDeterministicPolicy]:
        def make_policy():
            return MLPDeterministicPolicy(
                obs_space, action_space, spec.actor.encoder, spec.actor.norm_beta
            )

        actor = make_policy()
        actor.initialize_parameters(initializer_spec)

        behavior = actor
        if spec.behavior == "parameter_noise":
            behavior = make_policy()
            behavior.load_state_dict(actor.state_dict())

        target_actor = actor
        if spec.separate_target_policy:
            target_actor = make_policy()
            target_actor.load_state_dict(actor.state_dict())
        if spec.smooth_target_policy:
            target_actor = DeterministicPolicy.add_gaussian_noise(
                target_actor, noise_stddev=spec.target_gaussian_sigma
            )

        return actor, behavior, target_actor

    @staticmethod
    def _make_critic(
        obs_space: Box, action_space: Box, spec: DDPGCriticSpec, initializer_spec: dict
    ) -> Tuple[QValueEnsemble]:
        def make_critic():
            return MLPQValue(obs_space, action_space, spec.critic.encoder)

        def make_critic_ensemble():
            n_critics = 2 if spec.critic.double_q else 1
            critics = [make_critic() for _ in range(n_critics)]

            if spec.critic.parallelize:
                return ForkedQValueEnsemble(critics)
            return QValueEnsemble(critics)

        critics = make_critic_ensemble()
        critics.initialize_parameters(initializer_spec)
        target_critics = make_critic_ensemble()
        target_critics.load_state_dict(critics)
        return critics, target_critics
