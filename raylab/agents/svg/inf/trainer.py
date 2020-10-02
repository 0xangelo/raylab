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

    _name = "SVG(inf)"
    _policy_class = SVGInfTorchPolicy

    @override(Trainer)
    def validate_config(self, config: dict):
        """Assert configuration values are valid."""
        super().validate_config(config)
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
        assert (
            config["batch_mode"] == "complete_episodes"
        ), "SVG(inf) uses full rollouts"

    @override(Trainer)
    def after_init(self):
        super().after_init()
        set_policy_with_env_fn(self.workers, fn_type="reward")
