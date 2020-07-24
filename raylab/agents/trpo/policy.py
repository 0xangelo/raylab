"""TRPO policy implemented in PyTorch."""
import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils import override
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.pytorch.optim import build_optimizer
from raylab.pytorch.optim.hessian_free import conjugate_gradient
from raylab.pytorch.optim.hessian_free import hessian_vector_product
from raylab.pytorch.optim.hessian_free import line_search
from raylab.pytorch.utils import flat_grad
from raylab.utils.dictionaries import get_keys
from raylab.utils.explained_variance import explained_variance


class TRPOTorchPolicy(TorchPolicy):
    """Policy class for Trust Region Policy Optimization."""

    # pylint:disable=abstract-method
    dist_class = WrapStochasticPolicy

    @property
    @override(TorchPolicy)
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.trpo import TRPOTrainer

        return TRPOTrainer.options

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers.update(
            critic=build_optimizer(self.module.critic, self.config["torch_optimizer"])
        )
        return optimizers

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
        batch_tensors = self.lazy_tensor_dict(samples)
        info = {}

        info.update(self._update_actor(batch_tensors))
        info.update(self._update_critic(batch_tensors))
        info.update(self.extra_grad_info(batch_tensors))
        info.update(self.get_exploration_info())

        return {LEARNER_STATS_KEY: info}

    def _update_actor(self, batch_tensors):
        info = {}
        cur_obs, actions, advantages = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            Postprocessing.ADVANTAGES,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute Policy Gradient
        surr_loss = -(self.module.actor.log_prob(cur_obs, actions) * advantages).mean()
        pol_grad = flat_grad(surr_loss, self.module.actor.parameters())
        info["grad_norm(pg)"] = pol_grad.norm().item()

        # Compute Natural Gradient
        descent_step, cg_info = self._compute_descent_step(pol_grad, cur_obs)
        info["grad_norm(nat)"] = descent_step.norm().item()
        info.update(cg_info)

        # Perform Line Search
        if self.config["line_search"]:
            new_params, line_search_info = self._perform_line_search(
                pol_grad, descent_step, surr_loss, batch_tensors,
            )
            info.update(line_search_info)
        else:
            new_params = (
                parameters_to_vector(self.module.actor.parameters()) - descent_step
            )

        vector_to_parameters(new_params, self.module.actor.parameters())
        return info

    def _compute_descent_step(self, pol_grad, obs):
        """Approximately compute the Natural gradient using samples.

        This is based on the Fisher Matrix formulation as the hessian of the average
        entropy. For more information, see:
        https://en.wikipedia.org/wiki/Fisher_information#Matrix_form

        Args:
            pol_grad (Tensor): The vector to compute the Fisher vector product with.
            obs (Tensor): The observations to evaluate the policy in.
        """
        config = self.config
        params = list(self.module.actor.parameters())
        with torch.no_grad():
            ent_acts, _ = self.module.actor.sample(obs, (config["fvp_samples"],))

        def entropy():
            return self.module.actor.log_prob(obs, ent_acts).neg().mean()

        def fvp(vec):
            return hessian_vector_product(entropy(), params, vec)

        descent_direction, elapsed_iters, residual = conjugate_gradient(
            lambda x: fvp(x) + config["cg_damping"] * x,
            pol_grad,
            cg_iters=config["cg_iters"],
        )

        fisher_norm = pol_grad.dot(descent_direction)
        delta = config["delta"]
        scale = 0 if fisher_norm < 0 else torch.sqrt(2 * delta / (fisher_norm + 1e-8))

        descent_direction = descent_direction * scale
        return descent_direction, {"cg_iters": elapsed_iters, "cg_residual": residual}

    def _perform_line_search(self, pol_grad, descent_step, surr_loss, batch_tensors):
        expected_improvement = pol_grad.dot(descent_step).item()

        cur_obs, actions, old_logp, advantages = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.ACTION_LOGP,
            Postprocessing.ADVANTAGES,
        )

        @torch.no_grad()
        def f_barrier(params):
            vector_to_parameters(params, self.module.actor.parameters())
            new_logp = self.module.actor.log_prob(cur_obs, actions)
            surr_loss = self._compute_surr_loss(old_logp, new_logp, advantages)
            avg_kl = torch.mean(old_logp - new_logp)
            return surr_loss.item() if avg_kl < self.config["delta"] else np.inf

        new_params, expected_improvement, improvement = line_search(
            f_barrier,
            parameters_to_vector(self.module.actor.parameters()),
            descent_step,
            expected_improvement,
            y_0=surr_loss.item(),
            **self.config["line_search_options"],
        )
        improvement_ratio = (
            improvement / expected_improvement if expected_improvement else np.nan
        )
        info = {
            "expected_improvement": expected_improvement,
            "actual_improvement": improvement,
            "improvement_ratio": improvement_ratio,
        }
        return new_params, info

    @staticmethod
    def _compute_surr_loss(old_logp, new_logp, advantages):
        return -torch.mean(torch.exp(new_logp - old_logp) * advantages)

    def _update_critic(self, batch_tensors):
        info = {}
        mse = nn.MSELoss()

        cur_obs, value_targets, value_preds = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            Postprocessing.VALUE_TARGETS,
            SampleBatch.VF_PREDS,
        )

        for _ in range(self.config["val_iters"]):
            with self.optimizers.optimize("critic"):
                loss = mse(self.module.critic(cur_obs).squeeze(-1), value_targets)
                loss.backward()

        info["vf_loss"] = loss.item()
        info["explained_variance"] = explained_variance(
            value_targets.numpy(), value_preds.numpy()
        )
        return info

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):  # pylint:disable=unused-argument
        """Return statistics right after components are updated."""
        cur_obs, actions, old_logp, value_targets, value_preds = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.ACTION_LOGP,
            Postprocessing.VALUE_TARGETS,
            SampleBatch.VF_PREDS,
        )

        info = {
            "kl_divergence": torch.mean(
                old_logp - self.module.actor.log_prob(cur_obs, actions)
            ).item(),
            "entropy": torch.mean(-old_logp).item(),
            "perplexity": torch.mean(-old_logp).exp().item(),
            "explained_variance": explained_variance(
                value_targets.numpy(), value_preds.numpy()
            ),
            "grad_norm(critic)": nn.utils.clip_grad_norm_(
                self.module.critic.parameters(), float("inf")
            ).item(),
        }
        return info
