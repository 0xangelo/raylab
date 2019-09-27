# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from raylab.utils.adaptive_kl import AdaptiveKLCoeffSpec


class AdaptiveKLCoeffMixin:
    """Adds adaptive KL penalty as in PPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kl_coeff_spec = AdaptiveKLCoeffSpec(**self.config["kl_schedule"])

    def update_kl_coeff(self, kl_div):
        """
        Update KL penalty based on observed divergence between successive policies.
        """
        self._kl_coeff_spec.adapt(kl_div)

    @property
    def curr_kl_coeff(self):
        """Return current KL coefficient."""
        return self._kl_coeff_spec.curr_coeff
