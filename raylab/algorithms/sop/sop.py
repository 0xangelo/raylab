"""Trainer and configuration for SOP."""
from ray.rllib.utils.annotations import override

from raylab.algorithms import with_common_config
from raylab.algorithms.mixins import ParameterNoiseMixin
from raylab.algorithms.off_policy import GenericOffPolicyTrainer
from .sop_policy import SOPTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # Add gaussian noise to the action when calculating the Deterministic
        # Policy Gradient
        "smooth_target_policy": True,
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
            "name": "DDPGModule",
            "actor": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                # === SQUASHING EXPLORATION PROBLEM ===
                # Maximum l1 norm of the policy's output vector before the squashing
                # function
                "beta": 1.2,
            },
            "critic": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
            "torch_script": True,
        },
        # === Rollout Worker ===
        "num_workers": 0,
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        "exploration": None,
        # Options for parameter noise exploration
        "param_noise_spec": {
            "initial_stddev": 0.1,
            "desired_action_stddev": 0.2,
            "adaptation_coeff": 1.01,
        },
        # Whether to act greedly or exploratory, mostly for evaluation purposes
        "greedy": False,
        # Additive Gaussian i.i.d. noise to add to actions before squashing
        "exploration_gaussian_sigma": 0.3,
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
        "evaluation_config": {"greedy": True, "pure_exploration_steps": 0},
    }
)


class SOPTrainer(ParameterNoiseMixin, GenericOffPolicyTrainer):
    """Single agent trainer for Streamlined Off-Policy Algorithm."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SAC"
    _default_config = DEFAULT_CONFIG
    _policy = SOPTorchPolicy

    @override(GenericOffPolicyTrainer)
    def _init(self, config, env_creator):
        self._set_parameter_noise_callbacks(config)
        super()._init(config, env_creator)
