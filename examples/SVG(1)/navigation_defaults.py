"""Tune experiment configuration for SVG(1) on Navigation.

This can be run from the command line by executing
`raylab experiment 'SVG(1)' --local-dir <experiment dir>
    --config 'examples/SVG(1)/navigation_defaults.py' --stop timesteps_total 10000`
"""
from ray import tune  # pylint: disable=unused-import


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "Navigation",
        "env_config": tune.grid_search(
            [
                {"deceleration_zones": None},
                {"deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]}},
            ]
        ),
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 3e-4},
            "value": {"lr": 3e-4},
            "policy": {"lr": 3e-4},
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
                "layers": (64, 64),
                "activation": "ReLU",
                "input_dependent_scale": True,
                "initializer_options": {"name": "xavier_uniform"},
            },
            "value": {
                "layers": (64, 64),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "model": {
                "layers": (64, 64),
                "activation": "ReLU",
                "delay_action": True,
                "initializer_options": {"name": "xavier_uniform"},
            },
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 32,
        "timesteps_per_iteration": 200,
        # === Exploration ===
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 200,
        # === Evaluation ===
        "evaluation_interval": 5,
        "evaluation_num_episodes": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
