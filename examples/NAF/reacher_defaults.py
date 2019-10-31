"""Tune experiment configuration to test NAF in MujocoReacher."""
import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "MujocoReacher",
        "env_config": {"max_episode_steps": 50, "time_aware": True},
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": True,
        # === Replay Buffer ===
        "buffer_size": int(2e4),
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {
            "layers": [400, 300],
            "activation": "ELU",
            "initializer": "orthogonal",
            "initializer_options": {"gain": np.sqrt(2)},
        },
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # diag_gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        # full_gaussian: use gaussian action space noise where the precision matrix is
        #     given by the advantage function P matrix
        "exploration": "parameter_noise",
        "pure_exploration_steps": 500,
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 1,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
