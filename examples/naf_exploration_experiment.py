"""Tune experiment configuration to compare exploration options in NAF.

This can be run from the command line by executing
`python scripts/tune_experiment.py NAF --local-dir <experiment dir>
    --config examples/naf_exploration_experiment --stop timesteps_total 100000`
"""
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
        # === Exploration ===
        "exploration": tune.grid_search(["diag_gaussian", "parameter_noise"]),
        "diag_gaussian_stddev": 0.2,
        "pure_exploration_steps": 10000,
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        "horizon": 500,
        "seed": tune.grid_search(list(range(10))),
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
