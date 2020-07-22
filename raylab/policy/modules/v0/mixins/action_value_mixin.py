"""Support for modules with action value functions as critics."""
import torch.nn as nn
from ray.rllib.utils import override

from raylab.pytorch.nn import StateActionEncoder
from raylab.pytorch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge


BASE_CONFIG = {
    "double_q": False,
    "initializer_options": {"name": "xavier_uniform"},
    "encoder": {"units": (32, 32), "activation": "ReLU", "delay_action": True},
}


class ActionValueMixin:
    """Adds constructor for modules with action value functions.

    Since it is common to use clipped double Q-Learning, critic is implemented as
    a ModuleList of action-value functions.
    """

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config.get("critic", {}), False, ["encoder"])
        obs_size, act_size = obs_space.shape[0], action_space.shape[0]

        def make_critic():
            return ActionValueFunction.from_scratch(
                obs_size,
                act_size,
                initializer_options=config["initializer_options"],
                **config["encoder"]
            )

        n_critics = 2 if config["double_q"] else 1
        critics = nn.ModuleList([make_critic() for _ in range(n_critics)])
        target_critics = nn.ModuleList([make_critic() for _ in range(n_critics)])
        target_critics.load_state_dict(critics.state_dict())
        return {"critics": critics, "target_critics": target_critics}


class ActionValueFunction(nn.Module):
    """Neural network module emulating a Q value function."""

    def __init__(self, logits_module, value_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        logits = self.logits_module(obs, actions)
        return self.value_module(logits)

    @classmethod
    def from_scratch(
        cls, obs_size, act_size, initializer_options=None, **logits_kwargs
    ):
        """Create an action value function with new logits and value modules."""
        logits_module = StateActionEncoder(obs_size, act_size, **logits_kwargs)
        initializer_options = initializer_options or {}
        logits_module.apply(
            initialize_(
                activation=logits_kwargs.get("activation"), **initializer_options
            )
        )
        value_module = nn.Linear(logits_module.out_features, 1)
        return cls(logits_module, value_module)
