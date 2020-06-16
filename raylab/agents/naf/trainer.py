"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.off_policy import with_base_config

from .policy import NAFTorchPolicy


DEFAULT_CONFIG = with_base_config(
    {
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {},
        # === Optimization ===
        # PyTorch optimizer and options to use
        "torch_optimizer": {"type": "Adam", "lr": 3e-4},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "raylab.utils.exploration.ParameterNoise",
            # Options for parameter noise exploration
            "param_noise_spec": {
                "initial_stddev": 0.1,
                "desired_action_stddev": 0.2,
                "adaptation_coeff": 1.01,
            },
            "pure_exploration_steps": 1000,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    }
)


class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    # pylint: disable=attribute-defined-outside-init

    _name = "NAF"
    _default_config = DEFAULT_CONFIG
    _policy = NAFTorchPolicy
