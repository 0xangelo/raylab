"""Trainer and configuration for MAPO."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAPOTorchPolicy

DEFAULT_MODULE = {
    "type": "ModelBasedSAC",
    "model": {
        "network": {"units": (128, 128), "activation": "Swish"},
        "ensemble_size": 1,
        "input_dependent_scale": True,
        "parallelize": False,
        "residual": True,
    },
    "critic": {"double_q": True},
}


@trainer.configure
@trainer.config(
    "losses/grad_estimator",
    "SF",
    info="""\
    Gradient estimator for optimizing expectations. Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.config(
    "losses/lambda",
    0.0,
    info="""\
    Model KL regularization to avoid degenerate solutions (needs tuning)
    """,
)
@trainer.config(
    "losses/model_samples",
    4,
    info="""\
    Number of next states to sample from the model when calculating the
    model-aware deterministic policy gradient
    """,
)
@trainer.config(
    "losses", {}, info="Configurations for model, actor, and critic loss functions"
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("torch_optimizer/models", {"type": "Adam", "lr": 1e-3})
@trainer.config("model_training", TrainingSpec().to_dict(), info=TrainingSpec.__doc__)
@trainer.config("holdout_ratio", 0, override=True)
@trainer.config("max_holdout", 0, override=True)
@trainer.config("evaluation_config/explore", False, override=True)
@trainer.config("rollout_fragment_length", 25, override=True)
@trainer.config("batch_mode", "truncate_episodes", override=True)
@sac_config
class MAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    _name = "MAPO"
    _policy = MAPOTorchPolicy
