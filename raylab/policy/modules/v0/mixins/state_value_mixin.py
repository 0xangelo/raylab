"""Support for modules with state value functions as critics."""
import torch.nn as nn

from raylab.torch.nn import FullyConnected
from raylab.torch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge


BASE_CONFIG = {
    "target_vf": True,
    "initializer_options": {"name": "xavier_uniform"},
    "encoder": {"units": (400, 200), "activation": "Tanh"},
}


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
        config = deep_merge(
            BASE_CONFIG,
            config.get("critic", {}),
            False,
            ["encoder", "initializer_options"],
        )

        def make_vf():
            logits_mod = FullyConnected(
                in_features=obs_space.shape[0], **config["encoder"]
            )
            logits_mod.apply(
                initialize_(
                    activation=config["encoder"]["activation"],
                    **config["initializer_options"]
                )
            )
            value_mod = nn.Linear(logits_mod.out_features, 1)
            return nn.Sequential(logits_mod, value_mod)

        modules["critic"] = make_vf()
        if config["target_vf"]:
            modules["target_critic"] = make_vf()
            modules["target_critic"].load_state_dict(modules["critic"].state_dict())
        return modules
