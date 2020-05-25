from navigation_base import get_config as base_config
from ray import tune
from ray.rllib.utils import merge_dicts


def get_config():
    return merge_dicts(
        base_config(),
        {
            # === Environment ===
            "env_config": tune.grid_search(
                [
                    {"deceleration_zones": None},
                    {"deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]}},
                ]
            ),
            # === MAPO model training ===
            # Type of model-training to use. Possible types include
            # daml: policy gradient-aware model learning
            # mle: maximum likelihood estimation
            "model_loss": "DAML",
            # Number of next states to sample from the model when calculating the
            # model-aware deterministic policy gradient
            "num_model_samples": 4,
            # Gradient estimator for model-aware dpg. Possible types include:
            # score_function, pathwise_derivative
            "grad_estimator": tune.grid_search(["SF", "PD"]),
            # === Debugging ===
            # Whether to use the environment's true model to sample states
            "true_model": True,
        },
    )
