def get_config():
    from ray import tune

    return {
        "env": "IndustrialBenchmark-v0",
        "env_config": {
            "obs_type": "markovian",
            "max_episode_steps": 1000,
            "time_aware": False,
        },
        "dpg_loss": tune.grid_search("default acme".split()),
        "buffer_size": int(1e5),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "polyak": 0.995,
        "module": {
            "type": "DDPG",
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
            "pure_exploration_steps": 2000,
        },
        "learning_starts": 2000,
        "train_batch_size": 256,
        "rollout_fragment_length": 100,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 500,
        "evaluation_interval": 2,
        "evaluation_num_episodes": 10,
    }
