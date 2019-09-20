"""Tune experiment configuration SVG(inf).

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(inf)' --local-dir <experiment dir>
    --config examples/svg_inf_defaults.py --stop timesteps_total 100000`
"""
import math

import torch
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp",
        # Don't set 'done' at the end of the episode. Note that you still need to
        # set this if soft_horizon=True, unless your env is actually running
        # forever without returning done=True.
        "no_done_at_end": True,
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === RolloutWorker ===
        "sample_batch_size": 1000,
        "batch_mode": "complete_episodes",
        "horizon": 250,
        "seed": tune.grid_search(list(range(10))),
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
        # === Reward Function ===
        # Reward function in PyTorch, so that gradients can propagate back to the policy
        # parameters. Note: this should work with batches of states and actions.
        "reward_fn": tune.function(reward_fn),
    }


def reward_fn(state, action, next_state):
    reward_theta = (torch.cos(next_state[..., 2]) + 1.0) / 2.0
    reward_x = torch.cos((next_state[..., 0] / 2.4) * (math.pi / 2.0))
    return reward_theta * reward_x
