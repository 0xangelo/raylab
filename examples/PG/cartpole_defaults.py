"""Tune experiment configuration for PG on CartPoleSwingUp."""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"time_aware": True, "max_episode_steps": 250},
        # Don't set 'done' at the end of the episode. Note that you still need to
        # set this if soft_horizon=True, unless your env is actually running
        # forever without returning done=True.
        "no_done_at_end": False,
        # === PG ===
        # Learning rate
        "lr": 0.001,
        # Use PyTorch as backend
        "framework": "torch",
        # === RolloutWorker ===
        "num_workers": 0,
        "rollout_fragment_length": 1000,
        "batch_mode": "complete_episodes",
        # === Model ===
        "model": {
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "tanh",
            # Number of hidden layers for fully connected net
            "fcnet_hiddens": [64, 64],
            # For control envs, documented in ray.rllib.models.Model
            "free_log_std": True,
        },
    }
