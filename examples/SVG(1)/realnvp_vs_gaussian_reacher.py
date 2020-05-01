from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "MujocoReacher",
        "env_config": {"max_episode_steps": 50, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "critic": {"lr": 1e-3},
            "actor": {"lr": 3e-4},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {},
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Module ===
        "module": tune.grid_search(
            [
                {"name": "SVGModule", "torch_script": True},
                {
                    "name": "SVGRealNVPActor",
                    "torch_script": True,
                    "actor": {
                        "obs_encoder": {
                            "units": (32, 32),
                            "activation": "ELU",
                            "layer_norm": True,
                            "initializer_options": {"name": "xavier_uniform"},
                        },
                        "num_flows": 4,
                        "flow_mlp": {
                            "units": (24,) * 4,
                            "activation": "ELU",
                            "layer_norm": True,
                            "initializer_options": {"name": "xavier_uniform"},
                        },
                    },
                },
            ]
        ),
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        "evaluation_interval": 5,
    }
