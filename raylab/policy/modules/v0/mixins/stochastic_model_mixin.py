"""Support for modules with stochastic policies."""
from raylab.policy.modules.model.stochastic import build_ensemble
from raylab.policy.modules.model.stochastic import build_single
from raylab.policy.modules.model.stochastic import EnsembleSpec
from raylab.utils.dictionaries import deep_merge


BASE_CONFIG = EnsembleSpec().to_dict()


class StochasticModelMixin:
    """Adds constructor for modules with stochastic dynamics model."""

    # pylint:disable=too-few-public-methods

    def _make_model(self, obs_space, action_space, config):
        config.setdefault("model", {})
        config["model"].setdefault("ensemble_size", 0)
        spec = self.process_config(config)
        return self.build_models(obs_space, action_space, spec)

    @staticmethod
    def process_config(config) -> EnsembleSpec:
        """Fill in default configuration for models."""
        return EnsembleSpec.from_dict(
            deep_merge(BASE_CONFIG, config.get("model", {}), False)
        )

    @staticmethod
    def build_models(obs_space, action_space, spec):
        """Decide whether to return a single model or an ensemble."""
        if spec.ensemble_size == 0:
            return {"model": build_single(obs_space, action_space, spec)}

        return {"models": build_ensemble(obs_space, action_space, spec)}
