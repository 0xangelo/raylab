"""Tune experiment configuration to test SAC in MujocoReacher."""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "TimeLimitedEnv",
        "env_config": {
            "env_id": "MujocoReacher",
            "max_episode_steps": 50,
            "time_aware": True,
        },
        # === Replay Buffer ===
        "buffer_size": int(2e4),
        # === Exploration ===
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
