# pylint:disable=missing-module-docstring
from raylab.options import option


class OffPolicyMixin:
    """Mixin for off-policy agents."""

    def validate_config(self, config: dict):
        """Assert configuration values are valid."""
        super().validate_config(config)
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."

    @staticmethod
    def add_options(trainer_cls: type) -> type:
        cls = trainer_cls
        for opt in [
            option("rollout_fragment_length", default=1, override=True),
            option("num_workers", default=0, override=True),
            option("evaluation_config/explore", False, override=True),
        ]:
            cls = opt(cls)
        return cls
