"""Trainer and configuration for SVG(1)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGOneTorchPolicy


@trainer.config(
    "torch_optimizer/type", "Adam", info="Optimizer type for model, actor, and critic"
)
@trainer.config("torch_optimizer/model", {"lr": 1e-3})
@trainer.config("torch_optimizer/actor", {"lr": 1e-3})
@trainer.config("torch_optimizer/critic", {"lr": 1e-3})
@trainer.config(
    "vf_loss_coeff",
    1.0,
    info="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.config("max_grad_norm", 10.0, info="Clip gradient norms by this value")
@trainer.config(
    "max_is_ratio", 5.0, info="Clip importance sampling weights by this value"
)
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config(
    "kl_schedule",
    {"initial_coeff": 0},
    info="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
)
@trainer.config(
    "replay_kl",
    True,
    info="""\
    Whether to penalize KL divergence with the current policy or past policies
    that generated the replay pool.
    """,
)
@trainer.config("module", {"type": "SVGModule-v0"}, override=True)
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.StochasticActor", override=True
)
@trainer.config("exploration_config/pure_exploration_steps", 1000)
@trainer.config("evaluation_config/explore", False)
@OffPolicyTrainer.with_base_specs
class SVGOneTrainer(OffPolicyTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint:disable=attribute-defined-outside-init
    _name = "SVG(1)"
    _policy = SVGOneTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        self.get_policy().set_reward_from_config(config["env"], config["env_config"])

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))

    @override(OffPolicyTrainer)
    def _before_replay_steps(self, policy):
        if not self.config["replay_kl"]:
            policy.update_old_policy()
