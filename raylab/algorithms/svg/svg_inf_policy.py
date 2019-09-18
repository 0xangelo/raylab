"""SVG(inf) policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
from raylab.algorithms.svg.svg_module import ParallelDynamicsModel
from raylab.distributions import DiagMultivariateNormal
import raylab.modules as modules


class SVGInfTorchPolicy(TorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    ACTION_LOGP = "action_logp"

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        self.off_policy_optimizer = self._make_off_policy_optimizer()
        self.on_policy_optimizer = self._make_on_policy_optimizer()

        # Flag for off-policy learning
        self._off_policy_learning = False

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SVG(inf)"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_inf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @torch.no_grad()
    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        if self._off_policy_learning:
            loss, info = self.compute_joint_dynamics_value_loss(batch_tensors)
            self.off_policy_optimizer.zero_grad()
            loss.backward()
            self.off_policy_optimizer.step()
            self.update_targets()
        else:
            loss, info = self.compute_stochastic_value_gradient_loss(batch_tensors)
            self.on_policy_optimizer.zero_grad()
            loss.backward()
            self.on_policy_optimizer.step()

        return {LEARNER_STATS_KEY: info}

    # === NEW METHODS ===

    @staticmethod
    def _make_module(obs_space, action_space, config):
        module = nn.ModuleDict()

        policy_logits_module = modules.FullyConnected(
            in_features=obs_space.shape[0],
            units=config["modules"]["policy"]["layers"],
            activation=config["modules"]["policy"]["activation"],
        )
        policy_dist_params_module = modules.DiagMultivariateNormalParams(
            policy_logits_module.out_features, action_space.shape[0]
        )
        module["policy"] = nn.Sequential(
            policy_logits_module, policy_dist_params_module
        )

        model_logits_modules = [
            modules.StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                units=config["modules"]["model"]["layers"],
                activation=config["modules"]["model"]["activation"],
            )
            for _ in range(obs_space.shape[0])
        ]
        module["model"] = ParallelDynamicsModel(*model_logits_modules)

        value_logits_module = modules.FullyConnected(
            in_features=obs_space.shape[0],
            units=config["modules"]["value"]["layers"],
            activation=config["modules"]["value"]["activation"],
        )
        value_output = modules.ValueFunction(value_logits_module.out_features)
        module["value"] = nn.Sequential(value_logits_module, value_output)

        return module

    def _make_off_policy_optimizer(self):
        pass

    def _make_on_policy_optimizer(self):
        pass

    def compute_joint_dynamics_value_loss(self, batch_tensors):
        """Compute dynamics MLE loss and fitted value function loss."""

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute Stochatic Value Gradient loss given a full trajectory."""
