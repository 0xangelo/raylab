"""PyTorch implementation of Goldstone dynamics."""
import math

import torch
from ray.rllib.utils.annotations import override

from ..dynamics import dynamics
from .reward_function import TorchRewardFunction


class TorchDynamics(dynamics):
    """Utility class to provide a differentiable goldstone potential."""

    def __init__(self, number_steps, max_required_step, safe_zone):
        super().__init__(number_steps, max_required_step, safe_zone)
        self.number_steps = number_steps
        self.max_required_step = max_required_step

    @override(dynamics)
    def reward(self, phi_idx, position):  # pylint: disable=arguments-differ
        idx = torch.round(self._strongest_penality_abs_idx + phi_idx)
        idx = torch.where(idx < 0, idx + 2 * self._strongest_penality_abs_idx, idx)
        phi = (
            (-self._strongest_penality_abs_idx + idx) * 2 * math.pi / self.number_steps
        )
        return TorchRewardFunction(phi, self.max_required_step).reward(position)

    @override(dynamics)
    def state_transition(self, domain, phi_idx, system_response, position):

        old_domain = domain

        # (0) compute new domain
        domain = self._compute_domain(old_domain, position)
        # (1) if domain change: system_response <- advantageous
        system_response = torch.where(
            domain != old_domain, torch.ones_like(system_response), system_response
        )
        # (2) compute & apply turn direction
        phi_idx = phi_idx + self._compute_angular_step(
            domain, phi_idx, system_response, position
        )
        # (3) Update system response if necessary
        system_response = self._updated_system_response(phi_idx, system_response)
        # (4) apply symmetry
        phi_idx = self._apply_symmetry(phi_idx)
        # (5) if self._phi_idx == 0: reset internal state
        cond = (phi_idx == 0) & (position.abs() <= self._safe_zone)
        domain = torch.where(cond, torch.ones_like(domain), domain)
        phi_idx = torch.where(cond, torch.zeros_like(phi_idx), phi_idx)
        system_response = torch.where(
            cond, torch.ones_like(system_response), system_response
        )

        return domain, phi_idx, system_response

    @override(dynamics)
    def _compute_domain(self, domain, position):
        return torch.where(position.abs() <= self._safe_zone, domain, position.sign())

    @override(dynamics)
    def _compute_angular_step(self, domain, phi_idx, system_response, position):
        return torch.where(
            # cool down: when position close to zero
            position.abs() <= self._safe_zone,
            -phi_idx.sign(),  # cool down
            torch.where(
                phi_idx == -domain * self._strongest_penality_abs_idx,
                torch.zeros_like(phi_idx),
                system_response * position.sign(),
            ),
        )

    @override(dynamics)
    def _updated_system_response(self, phi_idx, system_response):
        return torch.where(
            phi_idx.abs() >= self._strongest_penality_abs_idx,
            torch.ones_like(system_response).neg(),
            system_response,
        )

    @override(dynamics)
    def _apply_symmetry(self, phi_idx):
        return torch.where(
            phi_idx.abs() < self._strongest_penality_abs_idx,
            phi_idx,
            2 * self._strongest_penality_abs_idx
            - (
                (phi_idx + (4 * self._strongest_penality_abs_idx))
                % (4 * self._strongest_penality_abs_idx)
            ),
        )
