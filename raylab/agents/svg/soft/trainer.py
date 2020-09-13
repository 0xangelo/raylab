"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option
from raylab.utils.replay_buffer import ReplayField

from .policy import SoftSVGTorchPolicy


TORCH_OPTIMIZERS = {
    "model": {"type": "Adam", "lr": 1e-3},
    "actor": {"type": "Adam", "lr": 1e-3},
    "critic": {"type": "Adam", "lr": 1e-3},
    "alpha": {"type": "Adam", "lr": 1e-3},
}


@configure
@option(
    "target_entropy",
    None,
    help="""
Target entropy to optimize the temperature parameter towards
If "auto", will use the heuristic provided in the SAC paper,
H = -dim(A), where A is the action space
""",
)
@option("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@option(
    "vf_loss_coeff",
    1.0,
    help="Weight of the fitted V loss in the joint model-value loss",
)
@option("max_is_ratio", 5.0, help="Clip importance sampling weights by this value")
@option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@option("module/type", "SoftSVG")
@option(
    "exploration_config/type",
    "raylab.utils.exploration.StochasticActor",
    override=True,
)
@option("exploration_config/pure_exploration_steps", 1000)
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
