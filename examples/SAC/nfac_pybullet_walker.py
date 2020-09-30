from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Walker2DBulletEnv-v0",
        "env_config": {"max_episode_steps": 1000, "time_aware": True},
        # === Entropy ===
        "target_entropy": None,
        # === Replay Buffer ===
        "buffer_size": int(1e6),
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        "module": {
            "type": "OffPolicyNFAC",
            "actor": {
                "conditional_prior": True,
                "obs_encoder": {"units": (256,), "activation": "ReLU"},
                "num_flows": 4,
                "conditional_flow": False,
                "flow": {
                    "type": "AffineCouplingTransform",
                    "transform_net": {
                        "type": "MLP",
                        "num_blocks": 0,
                        "activation": "ReLU",
                    },
                },
            },
            "critic": {"encoder": {"units": (256, 256)}},
            "entropy": {"initial_alpha": 0.05},
        },
        "compile_policy": True,
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 10000},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 10,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": True},
        },
    }
