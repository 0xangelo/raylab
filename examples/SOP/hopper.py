def get_config():
    return {
        "env": "Hopper-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        "buffer_size": int(1e5),
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "polyak": 0.995,
        "module": {
            "type": "DDPG",
            "initializer": {"name": "orthogonal"},
            "actor": {
                "smooth_target_policy": True,
                "target_gaussian_sigma": 0.3,
                "beta": 1.2,
                "encoder": {"units": (256, 256), "activation": "Swish"},
            },
            "critic": {
                "double_q": True,
                "encoder": {
                    "units": (256, 256),
                    "activation": "Swish",
                    "delay_action": True,
                },
            },
        },
        "exploration_config": {
            "type": "raylab.utils.exploration.GaussianNoise",
            "noise_stddev": 0.3,
            "pure_exploration_steps": 5000,
        },
        "learning_starts": 5000,
        "train_batch_size": 256,
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1000,
        "evaluation_interval": 5,
        "evaluation_num_episodes": 10,
    }
