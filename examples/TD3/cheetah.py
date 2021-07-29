def policy_config() -> dict:
    return {
        "buffer_size": int(1e6),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "module": {
            "type": "TD3",
            "initializer": {"name": "xavier_uniform"},
            "actor": {
                "separate_behavior": False,
                "target_gaussian_sigma": 0.3,
                "beta": 1.2,
                "encoder": {
                    "units": (256, 256),
                    "activation": "ReLU",
                },
            },
            "critic": {
                "double_q": True,
                "encoder": {
                    "units": (256, 256),
                    "activation": "ReLU",
                    "delay_action": False,
                },
            },
        },
        "batch_size": 256,
        "exploration_config": {
            "pure_exploration_steps": 10000,
        },
    }


def get_config():
    return {
        "env": "HalfCheetah-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        "policy": policy_config(),
        "learning_starts": 10000,  # Sync with pure exploration steps
        "rollout_fragment_length": 1,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1000,
        "evaluation_interval": 5,
    }
