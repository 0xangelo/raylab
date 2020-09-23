from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Navigation",
        "env_config": tune.grid_search(
            [
                {"deceleration_zones": None},
                {"deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]}},
            ]
        ),
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "optimizer_options": {
            "model": {"lr": 3e-4},
            "value": {"lr": 3e-4},
            "policy": {"lr": 3e-4},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.2,
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "actor": {
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                },
                "input_dependent_scale": False,
            },
            "critic": {
                "target_vf": True,
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                },
            },
            "model": {
                "residual": True,
                "input_dependent_scale": False,
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "delay_action": True,
                    "initializer_options": {"name": "xavier_uniform"},
                },
            },
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 32,
        "timesteps_per_iteration": 200,
        # === Exploration ===
        "exploration_config": {"pure_exploration_steps": 200},
        # === Evaluation ===
        "evaluation_interval": 5,
        "evaluation_num_episodes": 5,
    }
