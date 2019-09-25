"""SVG(inf) policy class using PyTorch."""
import itertools
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
import raylab.algorithms.svg.svg_module as svgm
import raylab.modules as mods
import raylab.utils.pytorch as torch_util


Transition = collections.namedtuple("Transition", "obs actions rewards next_obs dones")


class SVGInfTorchPolicy(TorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        self.off_policy_optimizer, self.on_policy_optimizer = self.optimizer()

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
        dist_params = self.module.policy(obs_batch)
        actions = self.module.policy_rsample(dist_params)
        logp = self.module.policy_logp(dist_params, actions)

        extra_fetches = {
            self.ACTION_LOGP: logp.cpu().numpy(),
            **{k: v.numpy() for k, v in dist_params.items()},
        }
        return actions.cpu().numpy(), state_batches, extra_fetches

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        if self._off_policy_learning:
            loss, info = self.compute_joint_model_value_loss(batch_tensors)
            self.off_policy_optimizer.zero_grad()
            loss.backward()
            info.update(self.extra_grad_info(batch_tensors))
            self.off_policy_optimizer.step()
            self.update_targets()
        else:
            episodes = [self._lazy_tensor_dict(s) for s in samples.split_by_episode()]
            loss, info = self.compute_stochastic_value_gradient_loss(episodes)
            self.on_policy_optimizer.zero_grad()
            loss.backward()
            info.update(self.extra_grad_info(batch_tensors))
            self.on_policy_optimizer.step()
            info.update(self.extra_apply_info(batch_tensors))

        return {LEARNER_STATS_KEY: info}

    @override(TorchPolicy)
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

    # === NEW METHODS ===

    def off_policy_learning(self, learn_off_policy):
        """Set the current learning state to off-policy or not."""
        self._off_policy_learning = learn_off_policy

    @staticmethod
    def _make_module(obs_space, action_space, config):
        module = nn.ModuleDict()

        model_config = config["module"]["model"]
        model_logits_modules = [
            mods.StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                units=model_config["layers"],
                activation=model_config["activation"],
            )
            for _ in range(obs_space.shape[0])
        ]
        module.model = svgm.ParallelDynamicsModel(*model_logits_modules)
        module.model.apply(
            torch_util.initialize_(
                model_config["initializer"], **model_config["initializer_options"]
            )
        )

        value_config = config["module"]["value"]

        def make_value_module():
            value_logits_module = mods.FullyConnected(
                in_features=obs_space.shape[0],
                units=value_config["layers"],
                activation=value_config["activation"],
            )
            value_output = mods.ValueFunction(value_logits_module.out_features)

            value_module = nn.Sequential(value_logits_module, value_output)
            value_module.apply(
                torch_util.initialize_(
                    value_config["initializer"], **value_config["initializer_options"]
                )
            )
            return value_module

        module.value = make_value_module()
        module.target_value = make_value_module()

        policy_config = config["module"]["policy"]
        policy_logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["layers"],
            activation=policy_config["activation"],
        )
        policy_dist_param_module = mods.DiagMultivariateNormalParams(
            policy_logits_module.out_features,
            action_space.shape[0],
            input_dependent_scale=policy_config["input_dependent_scale"],
        )
        module.policy = nn.Sequential(policy_logits_module, policy_dist_param_module)
        module.policy.apply(
            torch_util.initialize_(
                policy_config["initializer"], **policy_config["initializer_options"]
            )
        )

        module.policy_logp = svgm.DiagNormalLogProb()
        module.model_logp = svgm.DiagNormalLogProb()
        module.policy_rsample = svgm.DiagNormalRSample()
        module.model_rsample = svgm.DiagNormalRSample()
        module.entropy = svgm.DiagNormalEntropy()
        module.kl_div = svgm.DiagNormalKL()

        return module

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""
        # Add recurrent policy-model combination
        module = self.module
        module.rollout = svgm.ReproduceRollout(
            module.policy,
            module.model,
            module.policy_rsample,
            module.model_rsample,
            reward_fn,
        )

    def compute_joint_model_value_loss(self, batch_tensors):
        """Compute model MLE loss and fitted value function loss."""
        columns = [
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
        ]
        trans = Transition(*[batch_tensors[c] for c in columns])

        dist_params = self.module.model(trans.obs, trans.actions)
        residual = trans.next_obs - trans.obs
        mle_loss = self.module.model_logp(dist_params, residual).mean().neg()

        with torch.no_grad():
            targets = self.module.target_value(trans.next_obs).squeeze(-1)
            dist_params = self.module.policy(trans.obs)
            curr_logp = self.module.policy_logp(dist_params, trans.actions)
            is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])
            is_ratio = torch.clamp(is_ratio, max=self.config["max_is_ratio"])

        targets = torch.where(
            trans.dones, trans.rewards, trans.rewards + self.config["gamma"] * targets
        )
        values = self.module.value(trans.obs).squeeze(-1)
        value_loss = torch.mean(
            is_ratio * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        joint_loss = mle_loss + self.config["vf_loss_coeff"] * value_loss
        info = {
            "off_policy_loss": joint_loss.item(),
            "mle_loss": mle_loss.item(),
            "value_loss": value_loss.item(),
        }
        return joint_loss, info

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        polyak = self.config["polyak"]
        torch_util.update_polyak(self.module.value, self.module.target_value, polyak)

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
            }
        return fetches

    @torch.no_grad()
    def extra_apply_info(self, batch_tensors):
        """Add average KL divergence between new and old policies."""
        new_params = self.module.policy(batch_tensors[SampleBatch.CUR_OBS])
        kl_div = self.module.kl_div(batch_tensors, new_params).mean().item()
        return {"policy_kl_div": kl_div}
