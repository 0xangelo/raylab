"""Trainer and configuration for SVG(inf)."""
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.trainer import Trainer
from raylab.options import configure
from raylab.options import option

from .policy import SVGInfTorchPolicy


@configure
@option("rollout_fragment_length", default=1, override=True)
@option("batch_mode", "complete_episodes", override=True)
@option("num_workers", default=0, override=True)
@option("evaluation_config/explore", True)
class SVGInfTrainer(Trainer):
    """Single agent trainer for SVG(inf)."""

    # pylint:disable=abstract-method
    _name = "SVG(inf)"

    @override(Trainer)
    def get_policy_class(self, _):
        return SVGInfTorchPolicy

    @staticmethod
    @override(Trainer)
    def validate_config(config: dict):
        """Assert configuration values are valid."""
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
        assert (
            config["batch_mode"] == "complete_episodes"
        ), "SVG(inf) uses full rollouts"
        assert (
            config["learning_starts"] == 0
        ), "No point in having a warmup/exploration phase"

    def optimize_policy_backend(self):
        pass

    @override(Trainer)
    def after_init(self):
        set_policy_with_env_fn(self.workers, fn_type="reward")
        super().optimize_policy_backend()
