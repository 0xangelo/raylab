from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetahBulletEnv-v0",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "on_policy": {"type": "Adam", "lr": 3e-4},
            "off_policy": {"type": "Adam", "lr": 3e-4},
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
                    "units": (100, 100),
                    "activation": "Tanh",
                    "input_dependent_scale": True,
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "critic": {
                "encoder": {
                    "units": (400, 200),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "model": {
                "encoder": {
                    "units": (40, 40),
                    "activation": "Tanh",
                    "delay_action": True,
                    "initializer_options": {"name": "orthogonal"},
                },
            },
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 256,
        # === Evaluation ===
        "evaluation_interval": 5,
    }
