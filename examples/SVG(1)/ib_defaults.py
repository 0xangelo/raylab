def get_config():
    return {
        # === Environment ===
        "env": "IndustrialBenchmark",
        "env_config": {
            "reward_type": "classic",
            "action_type": "continuous",
            "observation": "markovian",
            "max_episode_steps": 1000,
            "time_aware": True,
        },
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Optimization ===
        # Weight of the fitted V loss in the joint model-value loss
        "vf_loss_coeff": 1.0,
        # Clip gradient norms by this value
        "max_grad_norm": float("inf"),
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.0,
            "desired_kl": 0.01,
            "adaptation_coeff": 2.0,
            "threshold": 1.5,
        },
        # === Optimization ===
        # PyTorch optimizer to use
        "torch_optimizer": "Adam",
        # Optimizer options for each component.
        # Valid keys include: 'model', 'value', and 'policy'
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # === Regularization ===
        # Whether to penalize KL divergence with the current policy or past policies
        # that generated the replay pool.
        "replay_kl": True,
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        "evaluation_interval": 5,
    }
