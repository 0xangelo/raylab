"""Trainer and configuration for SOP."""
from raylab.algorithms import with_common_config
from raylab.algorithms.sac.sac import SACTrainer
from .sop_policy import SOPTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # Add gaussian noise to the action when calculating the Deterministic
        # Policy Gradient
        "target_policy_smoothing": True,
        # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
        "target_gaussian_sigma": 0.3,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "critic": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Rollout Worker ===
        "num_workers": 0,
        # === Exploration ===
        # Additive Gaussian i.i.d. noise to add to actions before squashing
        "exploration_gaussian_sigma": 0.3,
        # Whether to add i.i.d. Gaussian noise to the policy network's output when
        # interacting with the environment
        "sampler_noise": True,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 1000,
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"sampler_noise": False, "pure_exploration_steps": 0},
    }
)


class SOPTrainer(SACTrainer):
    """Single agent trainer for Streamlined Off-Policy Algorithm."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SOP"
    _default_config = DEFAULT_CONFIG
    _policy = SOPTorchPolicy
