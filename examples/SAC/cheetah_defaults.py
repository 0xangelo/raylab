"""Tune experiment configuration to test SAC in HalfCheetah-v3.

This can be run from the command line by executing
`raylab experiment SAC --config examples/naf_exploration_experiment.py -s timesteps_total 100000`
"""
import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "TimeLimitedEnv",
        "env_config": {
            "env_id": "HalfCheetah-v3",
            "max_episode_steps": 250,
            "time_aware": False,
        },
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": True,
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Exploration ===
        "pure_exploration_steps": 5000,
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
                "input_dependent_scale": True,
            },
            "critic": {
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
                "delay_action": True,
            },
        },
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 20,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
