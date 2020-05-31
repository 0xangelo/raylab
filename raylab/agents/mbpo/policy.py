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
from raylab.envs import get_reward_fn
from raylab.envs import get_termination_fn
from raylab.losses import ModelEnsembleMLE


ModelSnapshot = collections.namedtuple("ModelSnapshot", "epoch loss state_dict")


class MBPOTorchPolicy(SACTorchPolicy):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.reward_fn = get_reward_fn(self.config["env"], self.config["env_config"])
        self.termination_fn = get_termination_fn(
            self.config["env"], self.config["env_config"]
        )

        models = self.module.models
        self.loss_model = ModelEnsembleMLE(models)
        num_elites = self.config["num_elites"]
        assert num_elites <= len(models), "Cannot have more elites than models"
        self.rng = np.random.default_rng(self.config["seed"])
        self.elite_models = self.rng.choice(models, size=num_elites, replace=False)

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

    @torch.no_grad()
    @override(SACTorchPolicy)
    def extra_grad_info(self, component):
        if component == "models":
            grad_norms = [
                torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()
                for m in self.module.models
            ]
            return {"grad_norm(models)": np.mean(grad_norms)}
        return super().extra_grad_info(component)

    def optimize_model(self, train_samples, eval_samples=None):
        """Update models with samples.

        Args:
            train_samples (SampleBatch): training data
            eval_samples (Optional[SampleBatch]): holdout data
        """
        snapshots = self._build_snapshots()
        dataloader = self._build_dataloader(train_samples)
        eval_tensors = self._lazy_tensor_dict(eval_samples) if eval_samples else None

        info, snapshots = self._train_model_epochs(dataloader, snapshots, eval_tensors)
        info.update(self._restore_models_and_set_elites(snapshots))

        info.update(self.extra_grad_info("models"))
        return self._learner_stats(info)

    def _build_snapshots(self):
        return [
            ModelSnapshot(epoch=0, loss=None, state_dict=copy.deepcopy(m.state_dict()))
            for m in self.module.models
        ]

    def _build_dataloader(self, train_samples):
        train_tensors = self._lazy_tensor_dict(train_samples)
        dataset = TensorDictDataset(
            {k: train_tensors[k] for k in self.loss_model.batch_keys}
        )
        return torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=self.config["model_batch_size"]
        )

    def _train_model_epochs(self, dataloader, snapshots, eval_tensors):
        info = {}
        max_grad_steps = self.config["max_model_steps"] or float("inf")
        grad_steps = 0
        start = time.time()
        for epoch in self._model_epochs():
            for minibatch in dataloader:
                with self.optimizer.models.optimize():
                    loss, _ = self.loss_model(minibatch)
                    loss.backward()
                grad_steps += 1
                if grad_steps >= max_grad_steps:
                    break

            if eval_tensors:
                with torch.no_grad():
                    _, eval_info = self.loss_model(eval_tensors)

                snapshots = self._update_snapshots(epoch, snapshots, eval_info)
                info.update(eval_info)

            if self._terminate_epoch(epoch, snapshots, start, grad_steps):
                break

        info["model_epochs"] = epoch + 1
        return info, snapshots

    def _model_epochs(self):
        max_model_epochs = self.config["max_model_epochs"]
        return range(max_model_epochs) if max_model_epochs else itertools.count()

    def _update_snapshots(self, epoch, snapshots, info):
        def update_snapshot(idx, snap):
            cur_loss = info[f"loss(models[{idx}])"]
            threshold = self.config["improvement_threshold"]
            if snap.loss is None or (snap.loss - cur_loss) / snap.loss > threshold:
                return ModelSnapshot(
                    epoch=epoch,
                    loss=cur_loss,
                    state_dict=copy.deepcopy(self.module.models[idx].state_dict()),
                )
            return snap

        return [update_snapshot(i, s) for i, s in enumerate(snapshots)]

    def _terminate_epoch(self, epoch, snapshots, start_time_s, model_steps):
        patience_epochs = self.config["patience_epochs"] or float("inf")
        max_time_s = self.config["max_model_train_s"] or float("inf")
        max_model_steps = self.config["max_model_steps"] or float("inf")

        return (
            time.time() - start_time_s >= max_time_s
            or epoch - max(s.epoch for s in snapshots) >= patience_epochs
            or model_steps >= max_model_steps
        )

    def _restore_models_and_set_elites(self, snapshots):
        info = {}
        for idx, snap in enumerate(snapshots):
            self.module.models[idx].load_state_dict(snap.state_dict)
            info[f"loss(model[{idx}])"] = snap.loss

        elite_idxs = np.argsort([s.loss for s in snapshots])[: len(self.elite_models)]
        info["loss(models[elites])"] = np.mean([snapshots[i].loss for i in elite_idxs])
        self.elite_models = [self.module.models[i] for i in elite_idxs]
        return info

    @torch.no_grad()
    def generate_virtual_sample_batch(self, samples):
        """Rollout model with latest policy.

        Produces samples for populating the virtual buffer, hence no gradient
        information is retained.

        If a transition is terminal, the next transition, if any, is generated from
        the initial state passed through `samples`.
        """
        virtual_samples = []
        obs = init_obs = self.convert_to_tensor(samples[SampleBatch.CUR_OBS])

        for _ in range(self.config["model_rollout_length"]):
            model = self.rng.choice(self.elite_models)

            action, _ = self.module.actor.sample(obs)
            next_obs, _ = model.sample(obs, action)
            reward = self.reward_fn(obs, action, next_obs)
            done = self.termination_fn(obs, action, next_obs)

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
            obs = torch.where(done.unsqueeze(-1), init_obs, next_obs)

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
