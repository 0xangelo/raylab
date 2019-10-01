"""Tune experiment configuration for SVG(1) on Navigation.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(1)' --local-dir <experiment dir>
    --config examples/svg_one_navigation_defaults.py --stop timesteps_total 10000`
"""
import numpy as np
from ray import tune  # pylint: disable=unused-import


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "Navigation",
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": tune.grid_search([1e-3, 3e-4])},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.2,
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "layers": (100, 100),
                "activation": "Tanh",
                "input_dependent_scale": tune.grid_search([True, False]),
                "initializer_options": {"name": "orthogonal"},
            },
            "value": {
                "layers": (200, 100),
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
            },
            "model": {
                "layers": (20, 20),
                "activation": "Tanh",
                "delay_action": True,
                "initializer_options": {"name": "orthogonal"},
            },
        },
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 100,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
