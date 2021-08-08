"""SVG(1) policy class using PyTorch."""
import contextlib
import warnings

import torch
from nnrl.optim import get_optimizer_class
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import Tensor, nn

from raylab.agents.svg import SVGTorchPolicy
from raylab.options import configure, option
from raylab.policy import AdaptiveKLCoeffMixin, EnvFnMixin, TorchPolicy, learner_stats
from raylab.policy.losses import OneStepSVG
from raylab.policy.off_policy import OffPolicyMixin, off_policy_options
from raylab.utils.replay_buffer import ReplayField


@configure
@off_policy_options
@option("optimizer/type", "Adam", help="Optimizer type for model, actor, and critic")
@option("optimizer/model", {"lr": 1e-3})
@option("optimizer/actor", {"lr": 1e-3})
@option("optimizer/critic", {"lr": 1e-3})
@option("max_grad_norm", 10.0, help="Clip gradient norms by this value")
@option(
    "replay_kl",
    True,
    help="""
    Whether to penalize KL divergence with the current policy or past policies
    that generated the replay pool.
    """,
)
@option(
    "kl_schedule",
    {"initial_coeff": 0},
    help="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
)
@option("module/type", default="SVG")
@option("exploration_config/type", "raylab.utils.exploration.StochasticActor")
@option("exploration_config/pure_exploration_steps", 1000)
class SVGOneTorchPolicy(OffPolicyMixin, AdaptiveKLCoeffMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint:disable=too-many-ancestors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_actor = OneStepSVG(
            self.module.model, self.module.actor, self.module.critic
        )
        self.loss_actor.gamma = self.config["gamma"]

        self.build_replay_buffer()

    def build_replay_buffer(self):
        super().build_replay_buffer()
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))

    @override(TorchPolicy)
    def compile(self):
        warnings.warn(f"{type(self).__name__} is incompatible with TorchScript")

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(SVGTorchPolicy)
    def _make_module(self, obs_space, action_space, config: dict):
        config["module"].setdefault("actor", {})
        config["module"]["actor"]["old_policy"] = config["replay_kl"]
        return super()._make_module(obs_space, action_space, config)

    @override(SVGTorchPolicy)
    def _make_optimizers(self):
        """PyTorch optimizer to use."""
        optimizers = super()._make_optimizers()
        options = self.config["optimizer"]
        cls = get_optimizer_class(options["type"], wrap=True)
        modules = {
            "model": self.module.model,
            "actor": self.module.actor,
            "critic": self.module.critic,
        }
        param_groups = [
            dict(params=mod.parameters(), **options[name])
            for name, mod in modules.items()
        ]
        optimizers["all"] = cls(param_groups)
        return optimizers

    @learner_stats
    @override(OffPolicyMixin)
    def learn_on_batch(self, samples: SampleBatch) -> dict:
        self.update_old_policy()
        info = super().learn_on_batch(samples)
        info.update(self.update_kl_coeff(samples))
        return info

    def update_old_policy(self):
        """Copy params of current policy into old one for future KL computation."""
        self.module.old_actor.load_state_dict(self.module.actor.state_dict())

    @override(OffPolicyMixin)
    def improve_policy(self, batch: TensorDict) -> dict:
        batch, info = self.add_truncated_importance_sampling_ratios(batch)

        with self.optimizers.optimize("all"):
            model_value_loss, stats = self.compute_joint_model_value_loss(batch)
            info.update(stats)
            model_value_loss.backward()

            with self.freeze_model_and_critic():
                svg_loss, stats = self.loss_actor(batch)
                info.update(stats)
                kl_loss = self.curr_kl_coeff * self._avg_kl_divergence(batch)
                (svg_loss + kl_loss).backward()

        info.update(self.extra_grad_info(batch))
        self._update_polyak()
        return info

    @contextlib.contextmanager
    def freeze_model_and_critic(self):
        """Disable gradients for model and critic."""
        self.module.model.requires_grad_(False)
        self.module.critic.requires_grad_(False)
        yield
        self.module.model.requires_grad_(True)
        self.module.critic.requires_grad_(True)

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        batch = self.lazy_tensor_dict(sample_batch)
        return self._avg_kl_divergence(batch).item()

    def _avg_kl_divergence(self, batch: TensorDict) -> Tensor:
        if self.config["replay_kl"]:
            logp = self.module.actor.log_prob(
                batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
            )
            return torch.mean(batch[SampleBatch.ACTION_LOGP] - logp)

        old_act, old_logp = self.module.old_actor.rsample(batch[SampleBatch.CUR_OBS])
        logp = self.module.actor.log_prob(batch[SampleBatch.CUR_OBS], old_act)
        return torch.mean(old_logp - logp)

    @torch.no_grad()
    def extra_grad_info(self, batch: TensorDict) -> dict:
        """Compute gradient norms and policy statistics."""
        grad_norms = {
            f"grad_norm({k})": nn.utils.clip_grad_norm_(
                getattr(self.module, k).parameters(), float("inf")
            )
            for k in "model actor critic".split()
        }
        policy_info = {
            "entropy": self.module.actor.log_prob(
                batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
            )
            .mean()
            .neg()
            .item(),
            "curr_kl_coeff": self.curr_kl_coeff,
        }
        return {**grad_norms, **policy_info}
