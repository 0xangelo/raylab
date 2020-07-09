"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import trainer
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


@trainer.config(
    "target_entropy",
    None,
    info="""\
Target entropy to optimize the temperature parameter towards
If "auto", will use the heuristic provided in the SAC paper,
H = -dim(A), where A is the action space
""",
)
@trainer.config("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@trainer.config(
    "vf_loss_coeff",
    1.0,
    info="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.config(
    "max_is_ratio", 5.0, info="Clip importance sampling weights by this value"
)
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("exploration_config", EXPLORATION_CONFIG, override=True)
@trainer.config("evaluation_config/explore", False, override=True)
@OffPolicyTrainer.with_base_specs
class SoftSVGTrainer(OffPolicyTrainer):
    """Single agent trainer for SoftSVG."""

    # pylint:disable=attribute-defined-outside-init
    _name = "SoftSVG"
    _policy = SoftSVGTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        self.get_policy().set_reward_from_config(config["env"], config["env_config"])

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))
