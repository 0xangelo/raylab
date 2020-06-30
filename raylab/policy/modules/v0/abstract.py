"""General module classes."""
import torch.nn as nn


class AbstractActorCritic(nn.ModuleDict):
    """Generic Actor-Critic architecture.

    Abstract module containing policy and value functions with no weight sharing.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_actor(obs_space, action_space, config))
        self.update(self._make_critic(obs_space, action_space, config))


class AbstractModelActorCritic(nn.ModuleDict):
    """Module containing env model, policy, and value functions.

    Generic model-based architecture with disjoint model, actor, and critic.
    """

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_model(obs_space, action_space, config))
        self.update(self._make_critic(obs_space, action_space, config))
        self.update(self._make_actor(obs_space, action_space, config))
