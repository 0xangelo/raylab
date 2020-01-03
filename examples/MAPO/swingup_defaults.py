"""Tune experiment configuration to test MAPO in CartPoleSwingUp."""


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "CartPoleSwingUp",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # decision_aware: policy gradient-aware model learning
        # mle: maximum likelihood estimation
        "model_loss": "decision_aware",
        # Type of the used p-norm of the distance between gradients.
        # Can be float('inf') for infinity norm.
        "norm_type": 2,
        # Number of initial next states to sample from the model when calculating the
        # model-aware deterministic policy gradient
        "num_model_samples": 1,
        # Length of the rollouts from each next state sampled
        "model_rollout_len": 1,
        # Gradient estimator for model-aware dpg. Possible types include:
        # score_function, pathwise_derivative
        "grad_estimator": "pathwise_derivative",
        # === Debugging ===
        # Whether to use the environment's true model to sample states
        "true_model": True,
        # Degrade the true model using a constant bias, i.e., by adding a constant
        # vector to the model's output
        "model_bias": None,
        # Degrade the true model using zero-mean gaussian noise
        "model_noise_sigma": None,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
        "target_gaussian_sigma": 0.2,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for model
        "model_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (128, 128),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "critic": {
                "units": (128, 128),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        "exploration": "gaussian",
        # Options for parameter noise exploration
        "param_noise_spec": {
            "initial_stddev": 0.1,
            "desired_action_stddev": 0.3,
            "adaptation_coeff": 1.01,
        },
        # Additive Gaussian i.i.d. noise to add to actions before squashing
        "exploration_gaussian_sigma": 0.3,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 5000,
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
