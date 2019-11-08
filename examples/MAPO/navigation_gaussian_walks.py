# pylint: disable=missing-docstring
from ray import tune

from navigation_base import get_config as base_config


def get_config():
    return {
        **base_config(),
        # === Environment ===
        "env_config": {
            "deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]},
            "num_random_walks": tune.grid_search([8, 16]),
        },
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
        "grad_estimator": tune.grid_search(["pathwise_derivative"]),
        # Whether to use the environment's true model to sample states
        "true_model": False,
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for model
        "model_optimizer": {
            "name": "Adam",
            "options": {
                "lr": 3e-4,
                # "weight_decay": 1e-3,
            },
        },
        # === Replay Buffer ===
        "buffer_size": int(5e4),
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (64,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "critic": {
                "units": (64,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
            "model": {
                "units": (10,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }
