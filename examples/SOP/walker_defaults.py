from ray import tune


GAUSSIAN_NOISE = {
    "type": "raylab.utils.exploration.GaussianNoise",
    "noise_stddev": 0.3,
    "pure_exploration_steps": 10000,
}

PARAMETER_NOISE = {
    "type": "raylab.utils.exploration.ParameterNoise",
    "param_noise_spec": {
        "initial_stddev": 0.1,
        "desired_action_stddev": 0.3,
        "adaptation_coeff": 1.01,
    },
    "pure_exploration_steps": 10000,
}


def get_config():
    return {
        # === Environment ===
        "env": "Walker2d-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "type": "DDPGModule",
            "actor": {
                "smooth_target_policy": True,
                "target_gaussian_sigma": 0.2,
                "beta": 1.2,
                "encoder": {
                    "units": (256, 256),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                    "layer_norm": False,
                },
            },
            "critic": {
                "double_q": False,
                "encoder": {
                    "units": (256, 256),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                    "delay_action": True,
                },
            },
        },
        # === Exploration Settings ===
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": tune.grid_search([GAUSSIAN_NOISE, PARAMETER_NOISE]),
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
    }
