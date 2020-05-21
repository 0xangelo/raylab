"""Action distribution for compatibility with RLlib's interface."""
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override


class WrapModuleDist(ActionDistribution):
    """Stores a nn.Module and inputs, delegation all methods to the module."""

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampled_logp = None

    @override(ActionDistribution)
    def sample(self):
        action, logp = self.model.actor.sample(**self.inputs)
        self._sampled_logp = logp
        return action, logp

    @override(ActionDistribution)
    def deterministic_sample(self):
        if hasattr(self.model.actor, "deterministic"):
            return self.model.actor.deterministic(**self.inputs)
        return self.model.actor(**self.inputs), None

    @override(ActionDistribution)
    def sampled_action_logp(self):
        return self._sampled_logp

    @override(ActionDistribution)
    def logp(self, x):
        return self.model.actor.log_prob(value=x, **self.inputs)

    @override(ActionDistribution)
    def entropy(self):
        return self.model.actor.entropy(**self.inputs)
