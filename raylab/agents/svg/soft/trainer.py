"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option
from raylab.utils.replay_buffer import ReplayField

from .policy import SoftSVGTorchPolicy


@configure
@option("evaluation_config/explore", False, override=True)
class SoftSVGTrainer(OffPolicyTrainer):
    """Single agent trainer for SoftSVG."""

    # pylint:disable=attribute-defined-outside-init
    _name = "SoftSVG"
    _policy = SoftSVGTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        set_policy_with_env_fn(self.workers, fn_type="reward")

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))
