"""Tune experiment configuration to compare exploration options in NAF.

This can be run from the command line by executing
`raylab experiment NAF --config examples/naf_exploration_experiment.py -s timesteps_total 100000`
"""
from ray import tune


GAUSSIAN_NOISE = {
    "type": "raylab.utils.exploration.GaussianNoise",
    "noise_stddev": 0.3,
    "pure_exploration_steps": 5000,
}

PARAMETER_NOISE = {
    "type": "raylab.utils.exploration.ParameterNoise",
    "param_noise_spec": {
        "initial_stddev": 0.1,
        "desired_action_stddev": 0.3,
        "adaptation_coeff": 1.01,
    },
    "pure_exploration_steps": 5000,
}


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {
            # Maximum l1 norm of the policy's output vector before the squashing function
            "beta": 1.2,
            "encoder": {
                "units": (128, 128),
                "activation": "ReLU",
                "layer_norm": True,
                "initializer_options": {"name": "xavier_uniform"},
            },
        },
        # === Optimization ===
        # PyTorch optimizer and options to use
        "optimizer": {"type": "Adam", "lr": 3e-4},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Rollout Worker ===
        "num_workers": 0,
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": tune.grid_search([GAUSSIAN_NOISE, PARAMETER_NOISE]),
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
    }
