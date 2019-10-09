"""SVG(inf) policy class using PyTorch."""
import itertools
import functools

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.algorithms.svg.svg_base_policy import SVGBaseTorchPolicy
import raylab.algorithms.svg.svg_module as svgm
import raylab.utils.pytorch as torch_util


class SVGInfTorchPolicy(SVGBaseTorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.off_policy_optimizer, self.on_policy_optimizer = self.optimizer()
        # Flag for off-policy learning
        self._off_policy_learning = False

    @staticmethod
    @override(SVGBaseTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(inf)"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_inf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGBaseTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        if self._off_policy_learning:
            batch_tensors, info = self.add_importance_sampling_ratios(batch_tensors)
            loss, _info = self.compute_joint_model_value_loss(batch_tensors)
            info.update(_info)
            self.off_policy_optimizer.zero_grad()
            loss.backward()
            info.update(self.extra_grad_info(batch_tensors))
            self.off_policy_optimizer.step()
            self.update_targets()
        else:
            episodes = [self._lazy_tensor_dict(s) for s in samples.split_by_episode()]
            loss, info = self.compute_stochastic_value_gradient_loss(episodes)
            kl_div = self._avg_kl_divergence(batch_tensors)
            loss = loss + kl_div * self.curr_kl_coeff
            self.on_policy_optimizer.zero_grad()
            loss.backward()
            info.update(self.extra_grad_info(batch_tensors))
            self.on_policy_optimizer.step()
            info.update(self.update_kl_coeff(samples))

        return self._learner_stats(info)

    @override(SVGBaseTorchPolicy)
    def optimizer(self):
        """PyTorch optimizers to use."""
        optim_cls = torch_util.get_optimizer_class(self.config["off_policy_optimizer"])
        params = itertools.chain(
            *[self.module[k].parameters() for k in ["model", "value"]]
        )
        off_policy_optim = optim_cls(
            params, **self.config["off_policy_optimizer_options"]
        )

        optim_cls = torch_util.get_optimizer_class(self.config["on_policy_optimizer"])
        on_policy_optim = optim_cls(
            self.module.policy.parameters(),
            **self.config["on_policy_optimizer_options"]
        )

        return off_policy_optim, on_policy_optim

    @override(SVGBaseTorchPolicy)
    def set_reward_fn(self, reward_fn):
        # Add recurrent policy-model combination
        module = self.module
        module.rollout = svgm.ReproduceRollout(
            module.policy,
            module.model,
            module.policy_rsample,
            module.model_rsample,
            reward_fn,
        )

    # ================================= NEW METHODS ====================================

    def set_off_policy(self, learn_off_policy):
        """Set the current learning state to off-policy or not."""
        self._off_policy_learning = learn_off_policy

    learn_off_policy = functools.partialmethod(set_off_policy, True)
    learn_on_policy = functools.partialmethod(set_off_policy, False)

    def compute_stochastic_value_gradient_loss(self, episodes):
        """Compute Stochatic Value Gradient loss given full trajectories."""
        total_ret = 0
        for episode in episodes:
            init_obs = episode[SampleBatch.CUR_OBS][0]
            actions = episode[SampleBatch.ACTIONS]
            next_obs = episode[SampleBatch.NEXT_OBS]

            rewards, _ = self.module.rollout(actions, next_obs, init_obs)
            total_ret += rewards.sum()

        avg_sim_return = total_ret / len(episodes)
        return -avg_sim_return, {"avg_sim_return": avg_sim_return.item()}

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):
        """Compute gradient norm for components. Also clips on-policy gradient."""
        if self._off_policy_learning:
            model_params = self.module.model.parameters()
            value_params = self.module.value.parameters()
            fetches = {
                "model_grad_norm": nn.utils.clip_grad_norm_(model_params, float("inf")),
                "value_grad_norm": nn.utils.clip_grad_norm_(value_params, float("inf")),
            }
        else:
            policy_params = self.module.policy.parameters()
            fetches = {
                "policy_grad_norm": nn.utils.clip_grad_norm_(
                    policy_params, max_norm=self.config["max_grad_norm"]
                ),
                "policy_entropy": self.module.entropy(batch_tensors).mean().item(),
                "curr_kl_coeff": self.curr_kl_coeff,
            }
        return fetches
