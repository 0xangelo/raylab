"""Policy for MBPO using PyTorch."""
import collections
import copy
import itertools
import time

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.utils.pytorch as ptu
from raylab.agents.sac import SACTorchPolicy
from raylab.envs.rewards import get_reward_fn
from raylab.losses import ModelEnsembleMLE


ModelSnapshot = collections.namedtuple("ModelSnapshot", "epoch loss state_dict")


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

    def optimize_model(self, train_samples, eval_samples):
        """Update models with samples.

        Args:
            train_samples (SampleBatch): training data
            eval_samples (SampleBatch): holdout data
        """
        train_tensors, eval_tensors = map(
            self._lazy_tensor_dict, (train_samples, eval_samples)
        )
        snapshots = self._build_snapshots()

        dataloader = self._build_dataloader(train_tensors)
        max_time_s = self.config["max_model_train_s"] or float("inf")
        start = time.time()
        for epoch in self._model_epochs():
            for minibatch in dataloader:
                with self.optimizer.models.optimize():
                    loss, _ = self.loss_model(minibatch)
                    loss.backward()

            if eval_samples.count > 0:
                with torch.no_grad():
                    _, info = self.loss_model(eval_tensors)

                snapshots, early_stop = self._update_snapshots(epoch, snapshots, info)
            else:
                early_stop = False

            if early_stop or time.time() - start >= max_time_s:
                break

        info.update(self._restore_models(snapshots))

        info["model_epochs"] = epoch
        info.update(self.extra_grad_info("models"))
        return self._learner_stats(info)

    def _build_snapshots(self):
        return [
            ModelSnapshot(
                epoch=0, loss=float("inf"), state_dict=copy.deepcopy(m.state_dict())
            )
            for m in self.module.models
        ]

    def _build_dataloader(self, train_tensors):
        dataset = TensorDictDataset(
            {k: train_tensors[k] for k in self.loss_model.batch_keys}
        )
        return torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=self.config["model_batch_size"]
        )

    def _model_epochs(self):
        max_model_epochs = self.config["max_model_epochs"]
        return range(max_model_epochs) if max_model_epochs else itertools.count()

    def _update_snapshots(self, epoch, snapshots, info):
        def update_snapshot(idx, snap):
            cur_loss = info[f"loss(model[{idx}])"]
            improvement = (snap.loss - cur_loss) / snap.loss
            if improvement > self.config["improvement_threshold"] or snap.loss is None:
                return ModelSnapshot(
                    epoch=epoch,
                    loss=cur_loss,
                    state_dict=copy.deepcopy(self.module.models[idx].state_dict()),
                )
            return snap

        new = [update_snapshot(i, s) for i, s in enumerate(snapshots)]
        early_stop = epoch - max(s.epoch for s in new) >= self.config["patience_epochs"]
        return new, early_stop

    def _restore_models(self, snapshots):
        info = {}
        for idx, snap in enumerate(snapshots):
            self.module.models[idx].load_state_dict(snap.state_dict)
            info[f"loss(model[{idx}])"] = snap.loss
        return info

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


class TensorDictDataset(torch.utils.data.Dataset):
    """Dataset wrapping a dict of tensors."""

    def __init__(self, tensor_dict):
        super().__init__()
        batch_size = next(iter(tensor_dict.values())).size(0)
        assert all(tensor.size(0) == batch_size for tensor in tensor_dict.values())
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensor_dict.items()}

    def __len__(self):
        return next(iter(self.tensor_dict.values())).size(0)
