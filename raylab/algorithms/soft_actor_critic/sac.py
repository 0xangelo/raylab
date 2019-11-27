"""Dummy."""
from raylab.algorithms import with_common_config
from raylab.algorithms.off_policy import GenericOffPolicyTrainer
from .sac_policy import SACTorchPolicy

DEFAULT_CONFIG = with_common_config(
    {
        "num_workers": 0,
        "buffer_size": int(1e5),
        "evaluation_config": {"mean_action_only": True},
        "pure_exploration_steps": 20,
        "module": {
            "policy": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": True,
            },
            "critic": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        "mean_action_only": False,
        "clipped_double_q": True,
        "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        "alpha_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        "target_entropy": None,
        "polyak": 0.995,
    }
)


class SACTrainer(GenericOffPolicyTrainer):
    _name = "BSAC"
    _default_config = DEFAULT_CONFIG
    _policy = SACTorchPolicy
