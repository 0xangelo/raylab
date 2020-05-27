"""Policy for MBPO using PyTorch."""
import collections

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.utils.pytorch as ptu
from raylab.agents.sac import SACTorchPolicy
from raylab.envs.rewards import get_reward_fn
from raylab.losses import ModelEnsembleMLE


class MBPOTorchPolicy(SACTorchPolicy):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.loss_model = ModelEnsembleMLE(self.module.models)
        self.reward_fn = get_reward_fn(self.config["env"], self.config["env_config"])
        self.random = np.random.default_rng(self.config["seed"])

    @staticmethod
    @override(SACTorchPolicy)
    def get_default_config():
        """Return the default config for MBPO."""
        # pylint:disable=cyclic-import
        from raylab.agents.mbpo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SACTorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = "models actor critics alpha".split()

        optims = {k: ptu.build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    def optimize_model(self, samples):
        """Update models with samples."""
        batch_tensors = self._lazy_tensor_dict(samples)

        with self.optimizer.models.optimize():
            loss, info = self.loss_model(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("models"))
        return self._learner_stats(info)

    @torch.no_grad()
    def generate_virtual_sample_batch(self, samples):
        """Rollout model with latest policy.

        This is for populating the virtual buffer, hence no gradient information is
        retained.
        """
        virtual_samples = []
        obs = self.convert_to_tensor(samples[SampleBatch.CUR_OBS])

        for _ in range(self.config["model_rollout_length"]):
            model = self.random.choice(self.module.models)

            action, _ = self.module.actor.sample(obs)
            next_obs, _ = model.sample(obs, action)
            reward = self.reward_fn(obs, action, next_obs)
            done = torch.zeros_like(reward).bool()

            transition = {
                SampleBatch.CUR_OBS: obs,
                SampleBatch.ACTIONS: action,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.REWARDS: reward,
                SampleBatch.DONES: done,
            }
            virtual_samples += [
                SampleBatch({k: v.numpy() for k, v in transition.items()})
            ]
            obs = next_obs

        return SampleBatch.concat_samples(virtual_samples)
