"""Action distribution for compatibility with RLlib's interface."""
import torch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils import override
from torch import Tensor

from .modules.actor.policy.deterministic import DeterministicPolicy
from .modules.actor.policy.stochastic import StochasticPolicy
from .modules.v0.mixins.stochastic_actor_mixin import StochasticPolicy as V0StochasticPi


class WrapStochasticPolicy(ActionDistribution):
    """Wraps an nn.Module with a stochastic actor and its inputs.

    Expects actor to be a StochasticPolicy instance.
    """

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self.model, "actor")
        assert isinstance(self.model.actor, (V0StochasticPi, StochasticPolicy))
        self._sampled_logp = None

    @override(ActionDistribution)
    def sample(self):
        action, logp = self.model.actor.sample(**self.inputs)
        self._sampled_logp = logp
        return action, logp

    @override(ActionDistribution)
    def deterministic_sample(self):
        return self.model.actor.deterministic(**self.inputs)

    @override(ActionDistribution)
    def sampled_action_logp(self):
        return self._sampled_logp

    @override(ActionDistribution)
    def logp(self, x):
        return self.model.actor.log_prob(value=x, **self.inputs)

    @override(ActionDistribution)
    def entropy(self):
        return self.model.actor.entropy(**self.inputs)


class WrapDeterministicPolicy(ActionDistribution):
    """Wraps an nn.Module with a deterministic actor and its inputs.

    Expects actor to be a DeterministicPolicy instance.
    """

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self.model, "actor") and isinstance(
            self.model.actor, DeterministicPolicy
        )
        assert hasattr(self.model, "behavior") and isinstance(
            self.model.behavior, DeterministicPolicy
        )

    @override(ActionDistribution)
    def sample(self):
        action = self.model.behavior(**self.inputs)
        return action, None

    def sample_inject_noise(self, noise_stddev: float) -> Tensor:
        """Add zero-mean Gaussian noise to the actions prior to normalizing them."""
        unconstrained_action = self.model.behavior.unconstrained_action(**self.inputs)
        unconstrained_action += torch.randn_like(unconstrained_action) * noise_stddev
        return self.model.behavior.squash_action(unconstrained_action), None

    @override(ActionDistribution)
    def deterministic_sample(self):
        return self.model.actor(**self.inputs), None

    @override(ActionDistribution)
    def sampled_action_logp(self):
        return None

    @override(ActionDistribution)
    def logp(self, x):
        return None
