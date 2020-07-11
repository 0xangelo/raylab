"""Trainer and configuration for MAPO."""
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.policy.model_based.training_mixin import DataloaderSpec
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

MODEL_TRAINING = TrainingSpec(
    dataloader=DataloaderSpec(batch_size=256),
    max_epochs=10,
    max_grad_steps=120,
    max_time=5,
    improvement_threshold=0.01,
    patience_epochs=5,
).to_dict()

TORCH_OPTIMIZERS = {
    "models": {"type": "Adam", "lr": 1e-3},
    "actor": {"type": "Adam", "lr": 1e-3},
    "critics": {"type": "Adam", "lr": 1e-3},
    "alpha": {"type": "Adam", "lr": 1e-3},
}

EXPLORATION_CONFIG = {"type": "raylab.utils.exploration.StochasticActor"}


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
    KL regularization to avoid degenerate solutions (needs tuning)
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
    "losses/true_model",
    False,
    info="Whether to use the environment's true model to sample states",
)
@trainer.config(
    "losses", {}, info="Configurations for model, actor, and critic loss functions"
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@trainer.config(
    "target_entropy",
    "auto",
    info="Target entropy to optimize the temperature parameter towards"
    " If 'auto', will use the heuristic provided in the SAC paper,"
    " H = -dim(A), where A is the action space",
)
@trainer.config("polyak", 0.995, info="Averaging factor for target networks")
@trainer.config("model_training", MODEL_TRAINING, info=TrainingSpec.__doc__)
@trainer.config("exploration_config", EXPLORATION_CONFIG, override=True)
@trainer.config("holdout_ratio", 0, override=True)
@trainer.config("max_holdout", 0, override=True)
@trainer.config("virtual_buffer_size", 0, override=True)
@trainer.config("model_rollouts", 0, override=True)
@trainer.config("real_data_ratio", 1, override=True)
@trainer.config("evaluation_config/explore", False, override=True)
@trainer.config("rollout_fragment_length", 25, override=True)
@trainer.config("batch_mode", "truncate_episodes", override=True)
@ModelBasedTrainer.with_base_specs
class MAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    # pylint:disable=attribute-defined-outside-init

    _name = "MAPO"
    _policy = MAPOTorchPolicy

    @staticmethod
    def validate_config(config):
        constants = {
            "holdout_ratio": 0,
            "max_holdout": 0,
            "virtual_buffer_size": 0,
            "model_rollouts": 0,
            "real_data_ratio": 1,
        }
        config.update(constants)
        ModelBasedTrainer.validate_config(config)

    @override(ModelBasedTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        policy = self.get_policy()
        if config["losses"]["true_model"]:
            worker = self.workers.local_worker()
            policy.set_dynamics_from_callable(worker.env.transition_fn)
