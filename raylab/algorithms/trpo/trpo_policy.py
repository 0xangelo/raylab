"""TRPO policy implemented in PyTorch."""
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as torch_util
from raylab.policy import TorchPolicy
from raylab.modules.catalog import get_module

from . import hf_util


class TRPOTorchPolicy(TorchPolicy):
    """Policy class for Trust Region Policy Optimization."""

    # pylint:disable=abstract-method
    ACTION_LOGP = "action_logp"

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default configuration for TRPO."""
        # pylint:disable=cyclic-import
        from raylab.algorithms.trpo.trpo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module = get_module(
            module_config["name"], obs_space, action_space, module_config
        )
        return torch.jit.script(module) if module_config["torch_script"] else module

    @override(TorchPolicy)
    def optimizer(self):
        return torch.optim.Adam(
            self.module.critic.parameters(), lr=self.config["val_lr"]
        )

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
        actions, logp = self.module.actor.sample(obs_batch)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches=other_agent_batches, episode=episode
        )

        last_obs = self.convert_to_tensor(sample_batch[SampleBatch.NEXT_OBS][-1])
        last_r = self.module.critic(last_obs).squeeze(-1).numpy()

        cur_obs = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[SampleBatch.VF_PREDS] = (
            self.module.critic(cur_obs).squeeze(-1).numpy()
        )
        sample_batch = compute_advantages(
            sample_batch,
            last_r,
            gamma=self.config["gamma"],
            lambda_=self.config["lambda"],
            use_gae=self.config["use_gae"],
        )
        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        info = {}

        cur_obs, actions, old_logp, advantages = _get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            self.ACTION_LOGP,
            Postprocessing.ADVANTAGES,
        )

        surr_loss = -(self.module.actor.log_prob(cur_obs, actions) * advantages).mean()
        pol_grad = torch_util.flat_grad(surr_loss, self.module.actor.parameters())
        info["pg_norm"] = pol_grad.norm().item()

        descent_step = self._compute_descent_step(
            pol_grad, cur_obs, self.module.actor, self.config
        )
        info["natural_pg_norm"] = descent_step.norm().item()
        if self.config["line_search"]:
            new_params, line_search_info = self._perform_line_search(
                pol_grad, descent_step, surr_loss, batch_tensors
            )
            info.update(line_search_info)
        else:
            new_params = (
                parameters_to_vector(self.module.actor.parameters()) - descent_step
            )
        vector_to_parameters(new_params, self.module.actor.parameters())

        info.update(self._fit_value_funtion(batch_tensors))
        with torch.no_grad():
            info["kl_divergence"] = torch.mean(
                old_logp - self.module.actor.log_prob(cur_obs, actions)
            ).item()
            info["entropy"] = torch.mean(-old_logp).item()
            info["perplexity"] = torch.mean(-old_logp).exp().item()
        return {LEARNER_STATS_KEY: info}

    @staticmethod
    def _compute_descent_step(pol_grad, cur_obs, actor, config):
        def fvp(vec):
            return hf_util.fisher_vec_prod(
                vec, cur_obs, actor, n_samples=config["fvp_samples"],
            )

        descent_direction = hf_util.conjugate_gradient(fvp, pol_grad)
        scale = torch.sqrt(
            2 * config["delta"] / (pol_grad.dot(descent_direction) + 1e-8)
        )
        return descent_direction * scale

    def _perform_line_search(self, pol_grad, descent_step, surr_loss, batch_tensors):
        expected_improvement = pol_grad.dot(descent_step).item()

        cur_obs, actions, old_logp, advantages = _get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            self.ACTION_LOGP,
            Postprocessing.ADVANTAGES,
        )

        @torch.no_grad()
        def f_barrier(params):
            vector_to_parameters(params, self.module.actor.parameters())
            new_logp = self.module.actor.log_prob(cur_obs, actions)
            surr_loss = self._compute_surr_loss(old_logp, new_logp, advantages)
            avg_kl = torch.mean(old_logp - new_logp)
            return surr_loss.item() if avg_kl < self.config["delta"] else float("inf")

        new_params, expected_improvement, improvement = hf_util.line_search(
            f_barrier,
            parameters_to_vector(self.module.actor.parameters()),
            descent_step,
            expected_improvement,
            y_0=surr_loss.item(),
        )
        info = {
            "expected_improvement": expected_improvement,
            "actual_improvement": improvement,
            "improvement_ratio": improvement / expected_improvement,
        }
        return new_params, info

    @staticmethod
    def _compute_surr_loss(old_logp, new_logp, advantages):
        return -torch.mean(torch.exp(new_logp - old_logp) * advantages)

    def _fit_value_funtion(self, batch_tensors):
        info = {}
        mse = torch.nn.MSELoss()

        cur_obs, value_targets, value_preds = _get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            Postprocessing.VALUE_TARGETS,
            SampleBatch.VF_PREDS,
        )

        for _ in range(self.config["val_iters"]):
            self._optimizer.zero_grad()
            loss = mse(self.module.critic(cur_obs).squeeze(-1), value_targets)
            loss.backward()
            self._optimizer.step()

        info["vf_loss"] = loss.item()
        info["explained_variance"] = explained_variance(
            value_targets.numpy(), value_preds.numpy()
        )
        return info


def explained_variance(targets, pred):
    """Compute the explained variance given targets and predictions."""
    # pylint:disable=invalid-name
    targets_var = np.var(targets, axis=0)
    diff_var = np.var(targets - pred, axis=0)
    return np.maximum(-1.0, 1.0 - (diff_var / targets_var))


def _get_keys(mapping, *keys):
    return (mapping[k] for k in keys)
