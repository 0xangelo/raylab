"""Trainer and configuration for SVG(1)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGOneTorchPolicy


@trainer.configure
@trainer.option(
    "torch_optimizer/type", "Adam", help="Optimizer type for model, actor, and critic"
)
@trainer.option("torch_optimizer/model", {"lr": 1e-3})
@trainer.option("torch_optimizer/actor", {"lr": 1e-3})
@trainer.option("torch_optimizer/critic", {"lr": 1e-3})
@trainer.option(
    "vf_loss_coeff",
    1.0,
    help="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.option("max_grad_norm", 10.0, help="Clip gradient norms by this value")
@trainer.option(
    "max_is_ratio", 5.0, help="Clip importance sampling weights by this value"
)
@trainer.option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@trainer.option(
    "kl_schedule",
    {"initial_coeff": 0},
    help="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
)
@trainer.option(
    "replay_kl",
    True,
    help="""
    Whether to penalize KL divergence with the current policy or past policies
    that generated the replay pool.
    """,
)
@trainer.option("module", {"type": "SVGModule-v0"}, override=True)
@trainer.option(
    "exploration_config/type", "raylab.utils.exploration.StochasticActor", override=True
)
@trainer.option("exploration_config/pure_exploration_steps", 1000)
@trainer.option("evaluation_config/explore", False)
class SVGOneTrainer(OffPolicyTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint:disable=attribute-defined-outside-init
    _name = "SVG(1)"
    _policy = SVGOneTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        set_policy_with_env_fn(self.workers, fn_type="reward")

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))

    @override(OffPolicyTrainer)
    def _before_replay_steps(self, policy):
        if not self.config["replay_kl"]:
            policy.update_old_policy()
