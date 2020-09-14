"""Trainer and configuration for SVG(1)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGOneTorchPolicy


@configure
@option(
    "replay_kl",
    True,
    help="""
    Whether to penalize KL divergence with the current policy or past policies
    that generated the replay pool.
    """,
)
@option("evaluation_config/explore", False)
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
