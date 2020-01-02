"""Tune experiment configuration to test MAPO in Reservoir."""
import numpy as np
from ray import tune
from raylab.envs.reservoir import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG


ENV_CONFIG = {"env": "Reservoir", **DEFAULT_ENV_CONFIG}

ACTOR_CRITIC_CONFIG = {
    "policy": {
        "units": (128, 128),
        "activation": "ELU",
        "initializer_options": {"name": "xavier_uniform", "gain": np.sqrt(2)},
    },
    "critic": {
        "units": (128, 128),
        "activation": "ELU",
        "initializer_options": {"name": "xavier_uniform", "gain": np.sqrt(2)},
        "delay_action": True,
    },
}

Q_LEARNING_CONFIG = {
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing function
    "beta": 1.2,
    # === Twin Delayed DDPG (TD3) tricks ===
    # Clipped Double Q-Learning: use the minimun of two target Q functions
    # as the next action-value in the target for fitted Q iteration
    "clipped_double_q": True,
    # Add gaussian noise to the action when calculating the Deterministic
    # Policy Gradient
    "target_policy_smoothing": True,
    # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
    "target_gaussian_sigma": 0.3,
    # Interpolation factor in polyak averaging for target networks.
    "polyak": 0.995,
}

REPLAY_CONFIG = {
    # === Replay Buffer ===
    "buffer_size": int(1e4)
}

EXPLORATION_CONFIG = {
    # === Exploration ===
    # Which type of exploration to use. Possible types include
    # None: use the greedy policy to act
    # parameter_noise: use parameter space noise
    # gaussian: use i.i.d gaussian action space noise independently for each
    #     action dimension
    "exploration": "gaussian",
    # Additive Gaussian i.i.d. noise to add to actions before squashing
    "exploration_gaussian_sigma": 0.3,
    # Until this many timesteps have elapsed, the agent's policy will be
    # ignored & it will instead take uniform random actions. Can be used in
    # conjunction with learning_starts (which controls when the first
    # optimization step happens) to decrease dependence of exploration &
    # optimization on initial policy parameters. Note that this will be
    # disabled when the action noise scale is set to 0 (e.g during evaluation).
    "pure_exploration_steps": 400,
}

EVALUATION_CONFIG = {
    # === Evaluation ===
    "evaluation_interval": 5,
    "evaluation_num_episodes": 5,
}

TRAINER_CONFIG = {
    # === RolloutWorker ===
    "sample_batch_size": 1,
    "batch_mode": "complete_episodes",
    # === Trainer ===
    "train_batch_size": 32,
    "timesteps_per_iteration": 400,
    # === Debugging ===
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level).
    "log_level": "WARN",
}


MAPO_CONFIG = {
    # === MAPO model training ===
    # Type of model-training to use. Possible types include
    # decision_aware: policy gradient-aware model learning
    # mle: maximum likelihood estimation
    "model_loss": tune.grid_search(["decision_aware", "mle"]),
    # Gradient estimator for model-aware dpg. Possible types include:
    # score_function, pathwise_derivative
    "grad_estimator": tune.grid_search(["score_function", "pathwise_derivative"]),
    # Type of the used p-norm of the distance between gradients.
    # Can be float('inf') for infinity norm.
    "norm_type": 2,
    # Number of next states to sample from the model when calculating the
    # model-aware deterministic policy gradient
    "num_model_samples": 8,
    # === Optimization ===
    # PyTorch optimizer to use for policy
    "policy_optimizer": {"name": "RMSprop", "options": {"lr": 1e-4}},
    # PyTorch optimizer to use for critic
    "critic_optimizer": {"name": "RMSprop", "options": {"lr": 1e-4}},
    # PyTorch optimizer to use for model
    "model_optimizer": {"name": "RMSprop", "options": {"lr": 1e-4}},
}


def get_config():  # pylint: disable=missing-docstring
    return {
        **Q_LEARNING_CONFIG,
        **REPLAY_CONFIG,
        **EXPLORATION_CONFIG,
        **EVALUATION_CONFIG,
        **TRAINER_CONFIG,
        "seed": tune.grid_search(list(range(4))),
        **ENV_CONFIG,
        **MAPO_CONFIG,
        "module": {
            **ACTOR_CRITIC_CONFIG,
            "model": {
                "units": (128, 128),
                "activation": "ELU",
                "initializer_options": {"name": "xavier_uniform", "gain": np.sqrt(2)},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }
