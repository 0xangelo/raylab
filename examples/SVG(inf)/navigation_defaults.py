"""Tune experiment configuration for SVG(inf) on Navigation.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(inf)' --local-dir <experiment dir>
    --config examples/svg_inf_navigation_defaults.py --stop timesteps_total 10000`
"""
from ray import tune


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "Navigation",
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "on_policy_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "on_policy_optimizer_options": {"lr": tune.grid_search([3e-4])},
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": tune.grid_search([0.0, 0.2]),
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {"input_dependent_scale": False},
            "model": {"delay_action": tune.grid_search([False])},
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 100,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
        "output": "logdir",
    }
