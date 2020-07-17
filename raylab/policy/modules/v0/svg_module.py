"""SVG Architecture with disjoint model, actor, and critic."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractModelActorCritic
from .mixins import StateValueMixin
from .mixins import StochasticActorMixin
from .mixins import SVGModelMixin


BASE_CONFIG = {
    "replay_kl": False,
    "actor": {
        "encoder": {"units": (100, 100), "activation": "Tanh"},
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    },
    "critic": {
        "target_vf": True,
        "encoder": {"units": (400, 200), "activation": "Tanh"},
        "initializer_options": {"name": "xavier_uniform"},
    },
    "model": {
        "residual": True,
        "input_dependent_scale": False,
        "encoder": {"units": (40, 40), "activation": "Tanh", "delay_action": True},
        "initializer_options": {"name": "xavier_uniform"},
    },
}


class SVGModule(
    SVGModelMixin, StochasticActorMixin, StateValueMixin, AbstractModelActorCritic,
):
    """Module architecture with reparameterized actor and model.

    Allows inference of noise variables given existing samples.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config, False, ["actor", "critic", "model"])
        super().__init__(obs_space, action_space, config)

        if config.get("replay_kl") is False:
            old = self._make_actor(obs_space, action_space, config)
            self.old_actor = old["actor"].requires_grad_(False)
