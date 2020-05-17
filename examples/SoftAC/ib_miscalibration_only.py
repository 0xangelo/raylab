# pylint: disable=missing-docstring
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "IndustrialBenchmark-v0",
        "env_config": {
            "setpoint": 50,
            "reward_type": "classic",
            "action_type": "continuous",
            "observation": "markovian",
            "operational_cost": False,
            "miscalibration": True,
            "fatigue": False,
            "auto_he": tune.grid_search([True, False]),
            "max_episode_steps": 500,
            "time_aware": True,
        },
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        "module": {
            "actor": {"encoder": {"units": (128, 128)}},
            "critic": {"encoder": {"units": (128, 128)}},
        },
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 2000},
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 1,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
    }
