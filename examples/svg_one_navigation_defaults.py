"""Tune experiment configuration for SVG(1) on Navigation.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(inf)' --local-dir <experiment dir>
    --config examples/svg_inf_defaults.py --stop timesteps_total 100000`
"""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Navigation",
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 10.0,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "layers": [100, 100],
                "activation": "Tanh",
                # "initializer": "xavier_normal",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
            },
            "value": {
                "layers": [400, 200],
                "activation": "Tanh",
                # "initializer": "xavier_normal",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
            },
            "model": {
                "layers": [40, 40],
                "activation": "Tanh",
                # "initializer": "xavier_normal",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
            },
        },
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
