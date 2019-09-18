"""SVG(inf) policy class using PyTorch."""
import itertools

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
from raylab.algorithms.svg.svg_module import ParallelDynamicsModel
from raylab.distributions import DiagMultivariateNormal
import raylab.modules as modules
import raylab.utils.pytorch as torch_util


class SVGInfTorchPolicy(TorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        # Currently hardcoded distributions
        self._model_dist = DiagMultivariateNormal
        self._policy_dist = DiagMultivariateNormal

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
        dist_params = self.module["policy"](obs_batch)
        dist = self._policy_dist(*dist_params)
        actions = dist.sample()
        actions_logp = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            state_batches,
            {self.ACTION_LOGP: actions_logp.cpu().numpy()},
        )

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
            loss, info = self.compute_joint_model_value_loss(batch_tensors)
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

        model_config = config["module"]["model"]
        model_logits_modules = [
            modules.StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                units=model_config["layers"],
                activation=model_config["activation"],
            )
            for _ in range(obs_space.shape[0])
        ]
        module["model"] = ParallelDynamicsModel(*model_logits_modules)
        module["model"].apply(
            torch_util.initialize_orthogonal(model_config["ortho_init_gain"])
        )

        value_config = config["module"]["value"]

        def make_value_module():
            value_logits_module = modules.FullyConnected(
                in_features=obs_space.shape[0],
                units=value_config["layers"],
                activation=value_config["activation"],
            )
            value_output = modules.ValueFunction(value_logits_module.out_features)

            value_module = nn.Sequential(value_logits_module, value_output)
            value_module.apply(
                torch_util.initialize_orthogonal(value_config["ortho_init_gain"])
            )
            return value_module

        module["value"] = make_value_module()
        module["target_value"] = make_value_module()

        policy_config = config["module"]["policy"]
        policy_logits_module = modules.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["layers"],
            activation=policy_config["activation"],
        )
        policy_dist_param_module = modules.DiagMultivariateNormalParams(
            policy_logits_module.out_features, action_space.shape[0]
        )
        module["policy"] = nn.Sequential(policy_logits_module, policy_dist_param_module)
        module["policy"].apply(
            torch_util.initialize_orthogonal(policy_config["ortho_init_gain"])
        )

        return module

    def _make_off_policy_optimizer(self):
        optim_cls = torch_util.get_optimizer_class(self.config["off_policy_optimizer"])
        off_policy_modules = self.module["model"], self.module["value"]
        params = itertools.chain(*[m.parameters() for m in off_policy_modules])
        return optim_cls(params, **self.config["off_policy_optimizer_options"])

    def _make_on_policy_optimizer(self):
        optim_cls = torch_util.get_optimizer_class(self.config["on_policy_optimizer"])
        params = self.module["policy"].parameters()
        return optim_cls(params, **self.config["on_policy_optimizer_options"])

    def compute_joint_model_value_loss(self, batch_tensors):
        """Compute model MLE loss and fitted value function loss."""
        # pylint: disable=too-many-locals
        obs, actions, rewards, next_obs, dones, old_logp = (
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.REWARDS],
            batch_tensors[SampleBatch.NEXT_OBS],
            batch_tensors[SampleBatch.DONES],
            batch_tensors[self.ACTION_LOGP],
        )

        dist_params = self.module["model"](obs, actions)
        dist = self._model_dist(*dist_params)
        mle_loss = dist.log_prob(next_obs).mean().neg()

        with torch.no_grad():
            next_val = self.module["target_value"](next_obs).squeeze(-1)
            dist_params = self.module["policy"](obs)
            curr_logp = self._policy_dist(*dist_params)
            is_ratio = torch.exp(curr_logp - old_logp)

        targets = torch.where(dones, rewards, rewards + self.config["gamma"] * next_val)
        values = self.module["value"](obs).squeeze(-1)
        value_loss = is_ratio * torch.nn.MSELoss(reduction="none")(values, targets)

        return mle_loss + self.config["vf_loss_coeff"] * value_loss.mean()

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute Stochatic Value Gradient loss given a full trajectory."""
