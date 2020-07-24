"""SOP policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import ActionDPG
from raylab.policy.losses import ClippedDoubleQLearning
from raylab.policy.losses import DeterministicPolicyGradient
from raylab.pytorch.nn.utils import update_polyak
from raylab.pytorch.optim import build_optimizer


class SOPTorchPolicy(TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._make_actor_loss()
        self.loss_critic = ClippedDoubleQLearning(
            self.module.critics, self.module.target_critics, self.module.target_actor,
        )
        self.loss_critic.gamma = self.config["gamma"]
        self._grad_step = 0

    @property
    @override(TorchPolicy)
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.sop import SOPTrainer

        return SOPTrainer.options

    def _make_actor_loss(self):
        if self.config["dpg_loss"] == "default":
            self.loss_actor = DeterministicPolicyGradient(
                self.module.actor, self.module.critics,
            )
        elif self.config["dpg_loss"] == "acme":
            self.loss_actor = ActionDPG(self.module.actor, self.module.critics)
            self.loss_actor.dqda_clipping = self.config["dqda_clipping"]
            self.loss_actor.clip_norm = self.config["clip_dqda_norm"]
        else:
            dpg_loss = self.config["dpg_loss"]
            raise ValueError(
                f"Invalid config for 'dpg_loss': {dpg_loss}."
                " Choose between 'default' and 'acme'"
            )

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]
        components = "actor critics".split()

        mapping = {
            name: build_optimizer(getattr(self.module, name), config[name])
            for name in components
        }

        optimizers.update(mapping)
        return optimizers

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)

        info = {}
        self._grad_step += 1
        info.update(self._update_critic(batch_tensors))
        if self._grad_step % self.config["policy_delay"] == 0:
            info.update(self._update_policy(batch_tensors))

        update_polyak(
            self.module.critics, self.module.target_critics, self.config["polyak"]
        )
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critics"):
            loss, info = self.loss_critic(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_policy(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            loss, info = self.loss_actor(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Return statistics right after components are updated."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                getattr(self.module, component).parameters(), float("inf")
            ).item()
        }

    @override(TorchPolicy)
    def get_weights(self):
        weights = super().get_weights()
        weights["grad_step"] = self._grad_step
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights):
        self._grad_step = weights["grad_step"]
        super().set_weights(weights)
