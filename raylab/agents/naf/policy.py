"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import ClippedDoubleQLearning
from raylab.torch.nn.utils import update_polyak
from raylab.torch.optim import build_optimizer


class NAFTorchPolicy(TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = ClippedDoubleQLearning(
            self.module.critics, self.module.target_vcritics
        )
        self.loss_fn.gamma = self.config["gamma"]

    @property
    @override(TorchPolicy)
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.naf import NAFTrainer

        return NAFTrainer.options

    @override(TorchPolicy)
    def _make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["type"] = "NAF"
        # pylint:disable=no-member
        return super()._make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers.update(
            naf=build_optimizer(self.module.critics, self.config["torch_optimizer"])
        )
        return optimizers

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)
        with self.optimizers.optimize("naf"):
            loss, info = self.loss_fn(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info())
        update_polyak(
            self.module.vcritics, self.module.target_vcritics, self.config["polyak"]
        )
        return info

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.critics.parameters(), float("inf")
            ).item()
        }
