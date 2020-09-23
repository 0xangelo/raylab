from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Navigation",
        # === Replay Buffer ===
        "buffer_size": int(1e4),
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
            "initial_coeff": tune.grid_search([0.0, 0.2]),
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "actor": {"input_dependent_scale": False},
            "model": {"encoder": {"delay_action": False}},
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 100,
    }
