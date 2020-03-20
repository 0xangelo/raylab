"""Generic model-based architecture with disjoint model, actor, and critic."""
import torch.nn as nn


class AbstractModelActorCritic(nn.ModuleDict):
    """Module containing env model, policy, and value functions."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_model(obs_space, action_space, config))
        self.update(self._make_critic(obs_space, action_space, config))
        self.update(self._make_actor(obs_space, action_space, config))
