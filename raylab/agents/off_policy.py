# pylint:disable=missing-module-docstring
from raylab.options import configure
from raylab.options import option

from .simple_trainer import SimpleTrainer


@configure
@option("rollout_fragment_length", default=1, override=True)
@option("num_workers", default=0, override=True)
@option("evaluation_config/explore", False, override=True)
class SimpleOffPolicy(SimpleTrainer):
    """Generic trainer for off-policy agents."""

    # pylint:disable=abstract-method
    @staticmethod
    def validate_config(config: dict):
        """Assert configuration values are valid."""
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
