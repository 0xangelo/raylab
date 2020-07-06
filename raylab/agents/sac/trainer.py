"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SACTorchPolicy


@trainer.config(
    "target_entropy",
    None,
    info="Target entropy to optimize the temperature parameter towards"
    " If 'auto', will use the heuristic provided in the SAC paper,"
    " H = -dim(A), where A is the action space",
)
@trainer.config("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3})
@trainer.config("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3})
@trainer.config("torch_optimizer/alpha", {"type": "Adam", "lr": 1e-3})
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config("module", {"type": "SAC", "critic": {"double_q": True}}, override=True)
@trainer.config(
    "exploration_config/type",
    "raylab.utils.exploration.StochasticActor",
    override=True,
)
@trainer.config("exploration_config/pure_exploration_steps", 1000)
@trainer.config("evaluation_config/explore", False, override=True)
@OffPolicyTrainer.with_base_specs
class SACTrainer(OffPolicyTrainer):
    """Single agent trainer for SAC."""

    _name = "SoftAC"
    _policy = SACTorchPolicy
