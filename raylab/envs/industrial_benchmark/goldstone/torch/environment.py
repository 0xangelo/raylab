"""Goldstone environment that uses PyTorch dynamics."""
from .dynamics import TorchDynamics


class TorchGSEnvironment:
    """Goldstone environment that uses PyTorch dynamics."""

    def __init__(self, number_steps, max_required_step, safe_zone):
        self._dynamics = TorchDynamics(number_steps, max_required_step, safe_zone)

    def reward(self, phi_idx, effective_shift):
        """Calculate reward."""
        return self._dynamics.reward(phi_idx, effective_shift)

    def state_transition(self, domain, phi_idx, system_response, effective_shift):
        """Calculate next goldstone variables."""
        domain, phi_idx, system_response = self._dynamics.state_transition(
            domain, phi_idx, system_response, effective_shift
        )
        return self.reward(phi_idx, effective_shift), domain, phi_idx, system_response
