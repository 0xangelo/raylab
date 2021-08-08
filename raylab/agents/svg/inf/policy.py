"""SVG(inf) policy class using PyTorch."""
import torch
from nnrl.optim import build_optimizer
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import nn

from raylab.agents.svg import SVGTorchPolicy
from raylab.options import configure, option
from raylab.policy import AdaptiveKLCoeffMixin, EnvFnMixin, learner_stats
from raylab.policy.losses import TrajectorySVG
from raylab.policy.off_policy import OffPolicyMixin, off_policy_options
from raylab.utils.replay_buffer import ReplayField


@configure
@off_policy_options
@option(
    "updates_per_step",
    1.0,
    help="Model and Value function updates per step in the environment",
)
@option("max_grad_norm", 10.0, help="Clip gradient norms by this value")
@option("optimizer/on_policy", {"type": "Adam", "lr": 1e-3})
@option("optimizer/off_policy", {"type": "Adam", "lr": 1e-3})
@option(
    "kl_schedule/",
    help="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
    allow_unknown_subkeys=True,
)
@option("module/type", default="SVG")
@option("exploration_config/type", "raylab.utils.exploration.StochasticActor")
class SVGInfTorchPolicy(OffPolicyMixin, AdaptiveKLCoeffMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint:disable=abstract-method,too-many-ancestors

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_actor = TrajectorySVG(
            self.module.model,
            self.module.actor,
            self.module.critic,
        )

        self.build_replay_buffer()

    @override(OffPolicyMixin)
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
        component_map = {
            "on_policy": self.module.actor,
            "off_policy": nn.ModuleList([self.module.model, self.module.critic]),
        }

        mapping = {
            name: build_optimizer(module, config[name])
            for name, module in component_map.items()
        }

        optimizers.update(mapping)
        return optimizers

    @override(SVGTorchPolicy)
    def compile(self):
        super().compile()
        self.loss_actor.compile()

    @override(OffPolicyMixin)
    def improve_policy(self, _):
        pass

    @learner_stats
    @override(SVGTorchPolicy)
    def learn_on_batch(self, samples: SampleBatch) -> dict:
        traj_len = samples.count
        self.add_to_buffer(samples)

        info = {}
        for _ in range(int(traj_len * self.config["updates_per_step"])):
            batch = self.replay.sample(self.config["batch_size"])
            batch = self.lazy_tensor_dict(batch)
            off_policy_stats = self._learn_off_policy(batch)

        info.update(off_policy_stats)
        info.update(self._learn_on_policy(samples))
        return info

    def _learn_off_policy(self, batch: TensorDict) -> dict:
        """Update off-policy components."""
        batch, info = self.add_truncated_importance_sampling_ratios(batch)

        with self.optimizers.optimize("off_policy"):
            loss, _info = self.compute_joint_model_value_loss(batch)
            info.update(_info)
            loss.backward()

        info.update(self.extra_grad_info(batch, on_policy=False))
        self._update_polyak()
        return info

    def _learn_on_policy(self, samples: SampleBatch) -> dict:
        """Update on-policy components."""
        batch = self.lazy_tensor_dict(samples)
        episodes = [self.lazy_tensor_dict(s) for s in samples.split_by_episode()]

        with self.optimizers.optimize("on_policy"):
            loss, info = self.loss_actor(episodes)
            kl_div = self._avg_kl_divergence(batch)
            loss = loss + kl_div * self.curr_kl_coeff
            loss.backward()

        info.update(self.extra_grad_info(batch, on_policy=True))
        info.update(self.update_kl_coeff(samples))
        return info

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch: SampleBatch):
        batch_tensors = self.lazy_tensor_dict(sample_batch)
        return self._avg_kl_divergence(batch_tensors).item()

    def _avg_kl_divergence(self, batch_tensors: TensorDict):
        logp = self.module.actor.log_prob(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        return torch.mean(batch_tensors[SampleBatch.ACTION_LOGP] - logp)

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors: TensorDict, on_policy: bool) -> dict:
        """Compute gradient norm for components. Also clips on-policy gradient."""
        if on_policy:
            params = self.module.actor.parameters()
            max_norm = self.config["max_grad_norm"]
            fetches = {
                "policy_grad_norm": nn.utils.clip_grad_norm_(params, max_norm),
                "policy_entropy": -batch_tensors[SampleBatch.ACTION_LOGP].mean(),
                "curr_kl_coeff": self.curr_kl_coeff,
            }
        else:
            max_norm = float("inf")
            fetches = {}
            params = self.module.model.parameters()
            fetches["model_grad_norm"] = nn.utils.clip_grad_norm_(params, max_norm)
            params = self.module.critic.parameters()
            fetches["value_grad_norm"] = nn.utils.clip_grad_norm_(params, max_norm)

        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in fetches.items()}
