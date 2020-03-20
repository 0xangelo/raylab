"""Support for modules with state value functions as critics."""
import torch.nn as nn

from raylab.modules import FullyConnected


class StateValueMixin:
    """Adds constructor for modules with state value functions.

    Its is not common to have target value functions in this setting, but the option
    is provided regardless.
    """

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        # pylint:disable=unused-argument
        modules = {}
        critic_config = config["critic"]

        def make_vf():
            logits_mod = FullyConnected(
                in_features=obs_space.shape[0],
                units=critic_config["units"],
                activation=critic_config["activation"],
                **critic_config["initializer_options"],
            )
            value_mod = nn.Linear(logits_mod.out_features, 1)
            return nn.Sequential(logits_mod, value_mod)

        modules["critic"] = make_vf()
        if critic_config["target_vf"]:
            modules["target_critic"] = make_vf()
            modules["target_critic"].load_state_dict(modules["critic"].state_dict())
        return modules
