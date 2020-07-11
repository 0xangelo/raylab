"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import ClippedDoubleQLearning
from raylab.policy.modules.critic.q_value import QValue
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.pytorch.nn.utils import update_polyak
from raylab.pytorch.optim import build_optimizer


class NAFValue(QValue):
    """Wrapper around NAF."""

    # pylint:disable=super-init-not-called,non-parent-init-called
    def __init__(self, naf: nn.Module):
        nn.Module.__init__(self)
        self.naf = naf

    def forward(self, obs, action):
        return self.naf(obs, action)


class TargetNAFValue(QValue):
    """Wrapper around NAF's state-value function."""

    # pylint:disable=super-init-not-called,non-parent-init-called
    def __init__(self, naf_value: nn.Module):
        nn.Module.__init__(self)
        self.naf_value = naf_value

    def forward(self, obs, action):
        return self.naf_value(obs)


class NAFTorchPolicy(TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        critics = QValueEnsemble([NAFValue(n) for n in self.module.critics])
        target_critics = QValueEnsemble(
            [TargetNAFValue(v) for v in self.module.target_vcritics]
        )
        self.loss_fn = ClippedDoubleQLearning(
            critics, target_critics, actor=lambda _: None
        )
        self.loss_fn.gamma = self.config["gamma"]

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.naf import NAFTrainer

        return NAFTrainer._default_config

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["type"] = "NAFModule-v0"
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
