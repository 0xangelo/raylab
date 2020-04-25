"""ACKTR policy implemented in PyTorch."""
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.policy import ACTION_LOGP
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.utils import hf_util
from raylab.utils.dictionaries import get_keys
from raylab.utils.explained_variance import explained_variance
from raylab.utils.kfac import KFACOptimizer
from raylab.modules.distributions import Normal
from raylab.policy import TorchPolicy


class ACKTRTorchPolicy(TorchPolicy):
    """Policy class for Actor-Critic with Kronecker factored Trust Region."""

    # pylint:disable=abstract-method

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default configuration for ACKTR."""
        # pylint:disable=cyclic-import
        from raylab.algorithms.acktr.acktr import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def optimizer(self):
        return KFACOptimizer(
            nn.ModuleList([self.module.actor, self.module.critic]),
            **self.config["kfac"],
        )

    @override(TorchPolicy)
    def compute_module_ouput(self, input_dict, state=None, seq_lens=None):
        return input_dict[SampleBatch.CUR_OBS], state

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

        cur_obs, actions, old_logp, advantages, value_targets = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            ACTION_LOGP,
            Postprocessing.ADVANTAGES,
            Postprocessing.VALUE_TARGETS,
        )

        # Compute whitening matrices
        self._compute_precond_matrices(cur_obs)

        # Compute joint loss
        self._optimizer.zero_grad()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr_loss = -(self.module.actor.log_prob(cur_obs, actions) * advantages).mean()
        info["loss(actor)"] = surr_loss.item()

        mse = nn.MSELoss()
        mse_loss = mse(self.module.critic(cur_obs).squeeze(-1), value_targets)
        info["loss(critic)"] = mse_loss.item()

        joint_loss = surr_loss + self.config["vf_loss_coeff"] * mse_loss
        joint_loss.backward()
        info.update(self.extra_grad_info(batch_tensors))
        pol_grad = [p.grad.clone() for p in self.module.actor.parameters()]
        self._optimizer.step()

        if self.config["line_search"]:
            info.update(self._perform_line_search(pol_grad, surr_loss, batch_tensors))

        with torch.no_grad():
            info["kl_divergence"] = torch.mean(
                old_logp - self.module.actor.log_prob(cur_obs, actions)
            ).item()
            info["entropy"] = torch.mean(-old_logp).item()
            info["perplexity"] = torch.mean(-old_logp).exp().item()
            info["explained_variance"] = explained_variance(
                value_targets.numpy(), batch_tensors[SampleBatch.VF_PREDS].numpy()
            )
        return self._learner_stats(info)

    def _compute_precond_matrices(self, cur_obs):
        n_samples = self.config["logp_samples"]
        optimizer = self._optimizer
        with optimizer.record_stats():
            optimizer.zero_grad()

            _, log_prob = self.module.actor.sample(cur_obs, (n_samples,))
            log_prob.mean().backward()

            fake_dist = Normal()
            values = self.module.critic(cur_obs).squeeze(-1)
            fake_samples = values + torch.randn_like(values)
            log_prob = fake_dist.log_prob(
                fake_samples.detach(), {"loc": values, "scale": torch.ones_like(values)}
            )
            log_prob.mean().backward()

    def _perform_line_search(self, pol_grad, surr_loss, batch_tensors):
        # pylint:disable=too-many-locals
        kl_clip = self._optimizer.state["kl_clip"]
        expected_improvement = sum(
            (g * p.grad.data).sum()
            for g, p in zip(pol_grad, self.module.actor.parameters())
        ).item()

        cur_obs, actions, old_logp, advantages = get_keys(
            batch_tensors,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            ACTION_LOGP,
            Postprocessing.ADVANTAGES,
        )

        @torch.no_grad()
        def f_barrier(scale):
            for par in self.module.actor.parameters():
                par.data.add_(scale, par.grad.data)
            new_logp = self.module.actor.log_prob(cur_obs, actions)
            for par in self.module.actor.parameters():
                par.data.sub_(scale, par.grad.data)
            surr_loss = self._compute_surr_loss(old_logp, new_logp, advantages)
            avg_kl = torch.mean(old_logp - new_logp)
            return surr_loss.item() if avg_kl < kl_clip else np.inf

        scale, expected_improvement, improvement = hf_util.line_search(
            f_barrier,
            1,
            1,
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
        for par in self.module.actor.parameters():
            par.data.add_(scale, par.grad.data)
        return info

    @staticmethod
    def _compute_surr_loss(old_logp, new_logp, advantages):
        return -torch.mean(torch.exp(new_logp - old_logp) * advantages)

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):  # pylint:disable=unused-argument
        """Return statistics right after components are updated."""
        return {
            f"grad_norm({k})": nn.utils.clip_grad_norm_(
                self.module[k].parameters(), float("inf")
            )
            for k in ("actor", "critic")
        }
