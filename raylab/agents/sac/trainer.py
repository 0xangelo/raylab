"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SACTorchPolicy


def sac_config(cls: type) -> type:
    """Add configurations for Soft Actor-Critic-based agents."""

    for config_setter in [
        trainer.option(
            "target_entropy",
            None,
            help="""Target entropy for temperature parameter optimization.

            If 'auto', will use the heuristic provided in the SAC paper,
            H = -dim(A), where A is the action space""",
        ),
        trainer.option("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3}),
        trainer.option("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3}),
        trainer.option("torch_optimizer/alpha", {"type": "Adam", "lr": 1e-3}),
        trainer.option(
            "polyak",
            0.995,
            help="Interpolation factor in polyak averaging for target networks.",
        ),
        trainer.option(
            "exploration_config/type",
            "raylab.utils.exploration.StochasticActor",
            override=True,
        ),
    ]:
        cls = config_setter(cls)

    return cls


@trainer.configure
@sac_config
@trainer.option("module", {"type": "SAC", "critic": {"double_q": True}}, override=True)
@trainer.option("exploration_config/pure_exploration_steps", 1000)
@trainer.option("evaluation_config/explore", False, override=True)
class SACTrainer(OffPolicyTrainer):
    """Single agent trainer for SAC."""

    _name = "SoftAC"
    _policy = SACTorchPolicy
