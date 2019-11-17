"""Tune experiment configuration to test MAPO in the Industrial Benchmark."""
from ray import tune
from ib_defaults import get_config as base_config


def get_config():  # pylint: disable=missing-docstring
    return {
        **base_config(),
        # === Environment ===
        "env": "IndustrialBenchmark",
        "env_config": {
            "reward_type": "classic",
            "action_type": "continuous",
            "markovian": True,
            "max_episode_steps": 500,
            "time_aware": True,
            "random_walks": {
                "num_walks": tune.grid_search([4, 8, 12]),
                "loc": 5.0,
                "scale": 2.0,
            },
        },
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # decision_aware: policy gradient-aware model learning
        # mle: maximum likelihood estimation
        "model_loss": tune.grid_search(["decision_aware", "mle"]),
        # Gradient estimator for model-aware dpg. Possible types include:
        # score_function, pathwise_derivative
        "grad_estimator": tune.grid_search(["score_function", "pathwise_derivative"]),
        # === Replay Buffer ===
        "buffer_size": int(2e4),
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
                "units": (22,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }
