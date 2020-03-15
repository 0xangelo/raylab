"""TRPO policy implemented in PyTorch."""
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
import raylab.utils.pytorch as torch_util
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
        module = nn.ModuleDict()

        return module

    @override(TorchPolicy)
    def optimizer(self):
        return torch.optim.Adam(
            self.module.value.parameters(), lr=self.config["val_lr"]
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
        actions, logp = self.module.sampler(obs_batch)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches=other_agent_batches, episode=episode
        )

        last_r = self.module.value(sample_batch[SampleBatch.NEXT_OBS][-1]).numpy()
        sample_batch[SampleBatch.VF_PREDS] = self.module.value(
            sample_batch[SampleBatch.CUR_OBS]
        ).numpy()
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

        cur_logp = self.module.policy.logp(batch_tensors[SampleBatch.CUR_OBS])
        surr_loss = -(
            (cur_logp - batch_tensors[self.ACTION_LOGP]).exp()
            * batch_tensors[Postprocessing.ADVANTAGES]
        ).mean()
        pol_grad = torch_util.flat_grad(surr_loss, self.module.policy.parameters())

        def fvp(vec):
            return hf_util.fisher_vec_prod(
                vec, batch_tensors[SampleBatch.CUR_OBS], self.module.policy
            )

        descent_direction = hf_util.conjugate_gradient(fvp, pol_grad)
        scale = torch.sqrt(
            2 * self.config["delta"] / (pol_grad.dot(descent_direction) + 1e-8)
        )
        descent_step = descent_direction * scale

        if self.config["line_search"]:
            new_params, line_search_info = self._perform_line_search(
                pol_grad, descent_step, surr_loss, batch_tensors
            )
            info.update(line_search_info)
        else:
            new_params = (
                parameters_to_vector(self.module.policy.parameters()) - descent_step
            )

        vector_to_parameters(new_params, self.module.policy.parameters())

        info.update(self._fit_value_funtion(batch_tensors))

        with torch.no_grad():
            info["kl_divergence"] = torch.mean(
                batch_tensors[self.ACTION_LOGP]
                - self.module.policy.logp(batch_tensors[SampleBatch.CUR_OBS])
            )
        info["explained_variance"] = explained_variance(
            batch_tensors[Postprocessing.VALUE_TARGETS],
            batch_tensors[SampleBatch.VF_PREDS],
            framework="torch",
        )
        return info

    def _perform_line_search(self, pol_grad, descent_step, surr_loss, batch_tensors):
        expected_improvement = pol_grad.dot(descent_step).item()

        @torch.no_grad()
        def f_barrier(params):
            vector_to_parameters(params, self.module.policy.parameters())
            new_logp = self.module.policy.logp(batch_tensors[SampleBatch.CUR_OBS])
            surr_loss = -(
                (new_logp - batch_tensors[self.ACTION_LOGP]).exp()
                * batch_tensors[Postprocessing.ADVANTAGES]
            ).mean()
            avg_kl = torch.mean(batch_tensors[self.ACTION_LOGP] - new_logp)
            return surr_loss.item() if avg_kl < self.config["delta"] else float("inf")

        new_params, expected_improvement, improvement = hf_util.line_search(
            f_barrier,
            parameters_to_vector(self.module.policy.parameters()),
            descent_step,
            expected_improvement,
            y_0=surr_loss.item(),
        )
        info = {
            "expected_improvement": expected_improvement,
            "actual_improment": improvement,
            "improvement_ratio": improvement / expected_improvement,
        }
        return new_params, info

    def _fit_value_funtion(self, batch_tensors):
        mse = torch.nn.MSELoss()
        for _ in range(self.config["val_iters"]):
            self._optimizer.zero_grad()
            loss = mse(
                self.module.value(batch_tensors[SampleBatch.CUR_OBS]),
                batch_tensors[Postprocessing.VALUE_TARGETS],
            ).backward()
            self._optimizer.step()
        return {"vf_loss": loss.item()}
