"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.losses import ClippedDoubleQLearning
from raylab.policy import TargetNetworksMixin
from raylab.policy import TorchPolicy
from raylab.pytorch.optim import build_optimizer


class NAFTorchPolicy(TargetNetworksMixin, TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_critics = [lambda s, _, v=v: v(s) for v in self.module.target_vcritics]
        self.loss_fn = ClippedDoubleQLearning(
            self.module.critics, target_critics, actor=lambda _: None,
        )
        self.loss_fn.gamma = self.config["gamma"]

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.agents.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["type"] = "NAFModule"
        module_config["double_q"] = config["clipped_double_q"]
        module_config["perturbed_policy"] = (
            config["exploration_config"]["type"]
            == "raylab.utils.exploration.ParameterNoise"
        )
        # pylint:disable=no-member
        return super().make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def make_optimizers(self):
        return {
            "naf": build_optimizer(self.module.critics, self.config["torch_optimizer"])
        }

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)
        with self.optimizers.optimize("naf"):
            loss, info = self.loss_fn(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info())
        self.update_targets("vcritics", "target_vcritics")
        return info

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.critics.parameters(), float("inf")
            ).item()
        }
