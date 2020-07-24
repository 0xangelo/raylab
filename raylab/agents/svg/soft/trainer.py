"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.utils.replay_buffer import ReplayField

from .policy import SoftSVGTorchPolicy


TORCH_OPTIMIZERS = {
    "model": {"type": "Adam", "lr": 1e-3},
    "actor": {"type": "Adam", "lr": 1e-3},
    "critic": {"type": "Adam", "lr": 1e-3},
    "alpha": {"type": "Adam", "lr": 1e-3},
}

DEFAULT_MODULE = {"type": "MaxEntModelBased-v0", "critic": {"target_vf": True}}
EXPLORATION_CONFIG = {
    "type": "raylab.utils.exploration.StochasticActor",
    "pure_exploration_steps": 1000,
}


@trainer.configure
@trainer.option(
    "target_entropy",
    None,
    help="""
Target entropy to optimize the temperature parameter towards
If "auto", will use the heuristic provided in the SAC paper,
H = -dim(A), where A is the action space
""",
)
@trainer.option("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@trainer.option(
    "vf_loss_coeff",
    1.0,
    help="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.option(
    "max_is_ratio", 5.0, help="Clip importance sampling weights by this value"
)
@trainer.option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@trainer.option("module", DEFAULT_MODULE, override=True)
@trainer.option("exploration_config", EXPLORATION_CONFIG, override=True)
@trainer.option("evaluation_config/explore", False, override=True)
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
