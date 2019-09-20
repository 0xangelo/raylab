"""SVG(inf) policy class using PyTorch."""
import itertools
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
from raylab.algorithms.svg.svg_module import (
    ParallelDynamicsModel,
    ReproduceRollout,
    NormalLogProb,
    NormalRSample,
)
import raylab.modules as modules
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
        actions = self.module["policy_rsample"](dist_params)
        logp = self.module["policy_logp"](dist_params, actions)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        if self._off_policy_learning:
            batch_tensors = self._lazy_tensor_dict(samples)
            loss, info = self.compute_joint_model_value_loss(batch_tensors)
            self.off_policy_optimizer.zero_grad()
            loss.backward()
            self.off_policy_optimizer.step()
            self.update_targets()
        else:
            episodes = [self._lazy_tensor_dict(s) for s in samples.split_by_episode()]
            loss, info = self.compute_stochastic_value_gradient_loss(episodes)
            self.on_policy_optimizer.zero_grad()
            loss.backward()
            params = self.module["policy"].parameters()
            nn.utils.clip_grad_norm_(params, max_norm=self.config["max_grad_norm"])
            self.on_policy_optimizer.step()

        return {LEARNER_STATS_KEY: info}

    # === NEW METHODS ===

    def off_policy_learning(self, learn_off_policy):
        """Set the current learning state to off-policy or not."""
        self._off_policy_learning = learn_off_policy

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

        module["policy_logp"] = NormalLogProb()
        module["model_logp"] = NormalLogProb()
        module["policy_rsample"] = NormalRSample()
        module["model_rsample"] = NormalRSample()

        # Add recurrent policy-model combination
        module["rollout"] = ReproduceRollout(
            module["policy"],
            module["model"],
            module["policy_rsample"],
            module["model_rsample"],
            config["reward_fn"],
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
        columns = [
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
        ]
        trans = Transition(*[batch_tensors[c] for c in columns])

        dist_params = self.module["model"](trans.obs, trans.actions)
        residual = trans.next_obs - trans.obs
        mle_loss = self.module["model_logp"](dist_params, residual).mean().neg()

        with torch.no_grad():
            next_val = self.module["target_value"](trans.next_obs).squeeze(-1)
            dist_params = self.module["policy"](trans.obs)
            curr_logp = self.module["policy_logp"](dist_params, trans.actions)
            is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])

        targets = torch.where(
            trans.dones, trans.rewards, trans.rewards + self.config["gamma"] * next_val
        )
        values = self.module["value"](trans.obs).squeeze(-1)
        value_loss = is_ratio * torch.nn.MSELoss(reduction="none")(values, targets) / 2

        joint_loss = mle_loss + self.config["vf_loss_coeff"] * value_loss.mean()
        return joint_loss, {}

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        module, target_module = self.module["value"], self.module["target_value"]
        torch_util.update_polyak(module, target_module, self.config["polyak"])

    def compute_stochastic_value_gradient_loss(self, episodes):
        """Compute Stochatic Value Gradient loss given full trajectories."""
        cur_obs = torch.stack([e[SampleBatch.CUR_OBS][0] for e in episodes])
        action_seqs = [e[SampleBatch.ACTIONS] for e in episodes]
        obs_seqs = [e[SampleBatch.NEXT_OBS] for e in episodes]
        action_batch = nn.utils.rnn.pad_sequence(action_seqs)
        obs_batch = nn.utils.rnn.pad_sequence(obs_seqs)

        rew_batch, last_obs = self.module["rollout"](action_batch, obs_batch, cur_obs)
        rew_seqs = [rew_batch[: len(o), i, ...] for i, o in enumerate(obs_seqs)]
        last_vals = [self.module["value"](o[None]).squeeze(0) for o in last_obs]
        gamma = torch.tensor(self.config["gamma"])  # pylint: disable=not-callable
        values = [
            torch.sum(r * gamma ** torch.arange(len(r)).float()) + v * gamma ** len(r)
            for r, v in zip(rew_seqs, last_vals)
        ]
        mean_value = sum(values) / len(values)
        return mean_value, {}
