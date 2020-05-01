from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Reacher-v2",
        "env_config": {"max_episode_steps": 50, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.5,
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "actor": {"encoder": {"initializer_options": {"name": "orthogonal"}}},
            "critic": {"encoder": {"initializer_options": {"name": "orthogonal"}}},
            "model": {"encoder": {"initializer_options": {"name": "orthogonal"}}},
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        "timesteps_per_iteration": 1000,
        # === Trainer ===
        "train_batch_size": 128,
        # === Evaluation ===
        "evaluation_interval": 1,
    }
