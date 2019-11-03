# pylint: disable=missing-docstring
from ray import tune

from navigation_base import get_config as base_config


def get_config():
    return {
        **base_config(),
        # === Environment ===
        "env_config": {"deceleration_zones": None},
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # decision_aware: policy gradient-aware model learning
        # mle: maximum likelihood estimation
        "model_loss": tune.grid_search(["decision_aware", "mle"]),
        # Type of the used p-norm of the distance between gradients.
        # Can be float('inf') for infinity norm.
        "norm_type": 2,
        # Number of next states to sample from the model when calculating the
        # model-aware deterministic policy gradient
        "num_model_samples": 4,
        # Gradient estimator for model-aware dpg. Possible types include:
        # score_function, pathwise_derivative
        "grad_estimator": tune.grid_search(["score_function", "pathwise_derivative"]),
        # Whether to use the environment's true model to sample states
        "true_model": False,
    }
