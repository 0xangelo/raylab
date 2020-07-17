def get_config():
    return {
        "env": "HalfCheetah-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        "dpg_loss": "acme",
        "buffer_size": int(2e5),
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "module": {
            "type": "DDPG",
            "initializer": {"name": "orthogonal"},
            "actor": {
                "separate_behavior": True,
                "smooth_target_policy": True,
                "target_gaussian_sigma": 0.3,
                "beta": 1.2,
                "encoder": {
                    "units": (256, 256),
                    "activation": "Swish",
                    "layer_norm": True,
                },
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
            "type": "raylab.utils.exploration.ParameterNoise",
            "param_noise_spec": {
                "initial_stddev": 0.1,
                "desired_action_stddev": 0.3,
                "adaptation_coeff": 1.01,
            },
            "pure_exploration_steps": 5000,
        },
        "learning_starts": 5000,
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        "evaluation_interval": 5,
    }
