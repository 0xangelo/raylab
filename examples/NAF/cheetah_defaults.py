from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetahBulletEnv-v0",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
        # === Optimization ===
        # PyTorch optimizer and options to use
        "optimizer": {"type": "Adam", "lr": 3e-4},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Network ===
        "module": {
            "encoder": {
                "units": (256, 256),
                "activation": "ELU",
                "layer_norm": True,
                "initializer_options": {"name": "orthogonal"},
            }
        },
        # === Exploration Settings ===
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            "type": "raylab.utils.exploration.ParameterNoise",
            "param_noise_spec": {
                "initial_stddev": 0.1,
                "desired_action_stddev": 0.3,
                "adaptation_coeff": 1.01,
            },
            "pure_exploration_steps": 10000,
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
    }
