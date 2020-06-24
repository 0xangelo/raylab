"""Trainer and configuration for SOP."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.off_policy import with_base_config

from .policy import SOPTorchPolicy


DEFAULT_CONFIG = with_base_config(
    {
        # === SOPTorchPolicy ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # PyTorch optimizers to use
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 1e-3},
            "critics": {"type": "Adam", "lr": 1e-3},
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # Update policy every this number of calls to `learn_on_batch`
        "policy_delay": 1,
        "module": {"type": "DDPGModule"},
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


class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for Streamlined Off-Policy Algorithm."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SOP"
    _default_config = DEFAULT_CONFIG
    _policy = SOPTorchPolicy
