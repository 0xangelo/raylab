# pylint: disable=missing-docstring
from ray import tune

from navigation_base import get_config as base_config


def get_config():
    return {
        **base_config(),
        # === Environment ===
        "env_config": tune.grid_search(
            [
                {"deceleration_zones": None},
                {"deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]}},
            ]
        ),
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # decision_aware: policy gradient-aware model learning
        # mle: maximum likelihood estimation
        "model_loss": "decision_aware",
        # Type of the used p-norm of the distance between gradients.
        # Can be float('inf') for infinity norm.
        "norm_type": 2,
        # Number of next states to sample from the model when calculating the
        # model-aware deterministic policy gradient
        "num_model_samples": 4,
        # Gradient estimator for model-aware dpg. Possible types include:
        # score_function, pathwise_derivative
        "grad_estimator": tune.grid_search(["score_function", "pathwise_derivative"]),
        # === Debugging ===
        # Whether to use the environment's true model to sample states
        "true_model": True,
        # Degrade the true model using a constant bias, i.e., by adding a constant
        # vector to the model's output
        "model_bias": None,
        # Degrade the true model using zero-mean gaussian noise
        "model_noise_sigma": None,
    }
