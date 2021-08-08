"""SoftSVG policy class using PyTorch."""
import torch
from nnrl.optim import build_optimizer
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import nn

from raylab.agents.svg import SVGTorchPolicy
from raylab.options import configure, option
from raylab.policy import EnvFnMixin
from raylab.policy.losses import ISSoftVIteration, MaximumEntropyDual, OneStepSoftSVG
from raylab.policy.off_policy import OffPolicyMixin, off_policy_options
from raylab.utils.replay_buffer import ReplayField


def default_optimizer() -> dict:
    # pylint:disable=missing-function-docstring
    return {
        "model": {"type": "Adam", "lr": 1e-3},
        "actor": {"type": "Adam", "lr": 1e-3},
        "critic": {"type": "Adam", "lr": 1e-3},
        "alpha": {"type": "Adam", "lr": 1e-3},
    }


@configure
@off_policy_options
@option(
    "target_entropy",
    None,
    help="""Target entropy to optimize the temperature parameter towards

    If "auto", will use the heuristic provided in the SAC paper,
    H = -dim(A), where A is the action space
    """,
)
@option("optimizer", default_optimizer(), override=True)
@option("module/type", "SoftSVG")
@option("exploration_config/type", "raylab.utils.exploration.StochasticActor")
@option("exploration_config/pure_exploration_steps", 1000)
class SoftSVGTorchPolicy(OffPolicyMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        module = self.module
        self.loss_actor = OneStepSoftSVG(
            module.model, module.actor, module.critic, module.alpha
        )
        self.loss_actor.gamma = self.config["gamma"]

        self.loss_critic = ISSoftVIteration(
            module.critic, module.target_critic, module.actor, module.alpha
        )
        self.loss_critic.gamma = self.config["gamma"]

        target_entropy = (
            -action_space.shape[0]
            if self.config["target_entropy"] == "auto"
            else self.config["target_entropy"]
        )
        self.loss_alpha = MaximumEntropyDual(
            module.alpha, module.actor.sample, target_entropy
        )

        self.build_replay_buffer()

    def build_replay_buffer(self):
        super().build_replay_buffer()
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(SVGTorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["optimizer"]
        module = self.module
        components = {
            "model": module.model,
            "actor": module.actor,
            "critic": module.critic,
            "alpha": module.alpha,
        }

        mapping = {
            name: build_optimizer(module, config[name])
            for name, module in components.items()
        }

        optimizers.update(mapping)
        return optimizers

    @torch.no_grad()
    @override(SVGTorchPolicy)
    def add_truncated_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        batch = batch_tensors
        curr_logp = self.module.actor.log_prob(
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
        )

        is_ratios = torch.exp(curr_logp - batch[SampleBatch.ACTION_LOGP])
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])

        batch[self.loss_actor.IS_RATIOS] = _is_ratios
        batch[self.loss_critic.IS_RATIOS] = _is_ratios

        info = {
            "is_ratio_max": is_ratios.max().item(),
            "is_ratio_mean": is_ratios.mean().item(),
            "is_ratio_min": is_ratios.min().item(),
            "cross_entropy": -curr_logp.mean().item(),
        }
        return batch, info

    @override(OffPolicyMixin)
    def improve_policy(self, batch: TensorDict) -> dict:
        batch, info = self.add_truncated_importance_sampling_ratios(batch)

        info.update(self._update_model(batch))
        info.update(self._update_critic(batch))
        info.update(self._update_actor(batch))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch))

        self._update_polyak()
        return info

    def _update_model(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("model"):
            model_loss, info = self.loss_model(batch)
            model_loss.backward()

        info.update(self.extra_grad_info("model"))
        return info

    def _update_critic(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("critic"):
            value_loss, info = self.loss_critic(batch)
            value_loss.backward()

        info.update(self.extra_grad_info("critic"))
        return info

    def _update_actor(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("actor"):
            svg_loss, info = self.loss_actor(batch)
            svg_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    def _update_alpha(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("alpha"):
            alpha_loss, info = self.loss_alpha(batch)
            alpha_loss.backward()

        info.update(self.extra_grad_info("alpha"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component: str) -> dict:
        """Return gradient statistics for component."""
        fetches = {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                getattr(self.module, component).parameters(), float("inf")
            ).item()
        }
        return fetches
