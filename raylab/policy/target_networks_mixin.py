# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from raylab.utils.pytorch import update_polyak


class TargetNetworksMixin:
    """Adds method to update target networks by name."""

    # pylint: disable=too-few-public-methods

    def update_targets(self, module, target_module):
        """Update target networks through one step of polyak averaging.

        Arguments:
            module (str): name of primary module in the policy's module dict
            target_module (str): name of target module in the policy's module dict
        """
        update_polyak(
            self.module[module], self.module[target_module], self.config["polyak"]
        )
