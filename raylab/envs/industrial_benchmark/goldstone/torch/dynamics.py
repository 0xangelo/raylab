"""PyTorch implementation of Goldstone dynamics."""
import math

import torch
from ray.rllib.utils.annotations import override

from ..dynamics import dynamics


class TorchDynamics(dynamics):
    """Utility class to provide a differentiable goldstone potential."""

    def __init__(self, number_steps, max_required_step, safe_zone):
        super().__init__(number_steps, max_required_step, safe_zone)
        self.number_steps = number_steps
        self.max_required_step = max_required_step
        self.alpha = 0.5849
        self.beta = 0.2924
        self.kappa = -0.6367

    @override(dynamics)
    def reward(self, phi_idx, effective_shift):  # pylint: disable=arguments-differ
        rho_s = torch.sin(math.pi * phi_idx / 12)
        omega = self.omega(rho_s, effective_shift)

        return (
            -self.alpha * omega ** 2
            + self.beta * omega ** 4
            + self.kappa * rho_s * omega
        )

    def omega(self, rho_s, effective_shift):
        """Compute omega as given by Equation (40)."""
        # pylint:disable=invalid-name
        varrho = rho_s.sign()
        r_opt = varrho * torch.max(rho_s.abs(), torch.as_tensor(2 * self._safe_zone))
        q = self.kappa * rho_s.abs() / (8 * self.beta)

        mask = q < -math.sqrt(1 / 27)
        r_min = torch.empty_like(r_opt)
        r_min[mask] = self._compute_r_min1(q[mask], varrho[mask])
        r_min[~mask] = self._compute_r_min2(q[~mask], varrho[~mask])

        mask = effective_shift.abs() <= r_opt.abs()
        omega = torch.empty_like(effective_shift)
        omega[mask] = self._compute_omega1(
            r_min[mask], r_opt[mask], effective_shift[mask]
        )
        omega[~mask] = self._compute_omega2(
            r_min[~mask], r_opt[~mask], effective_shift[~mask]
        )
        return omega

    @staticmethod
    def _compute_r_min1(q, varrho):
        """Compute r_min resulting from the first branch of Equation (44)."""
        # pylint:disable=invalid-name
        u = (-varrho * q + torch.sqrt(q ** 2 - (1 / 27))) ** (1 / 3)
        return (u + 1) / (3 * u)

    @staticmethod
    def _compute_r_min2(q, varrho):
        """Compute r_min resulting from the second branch of Equation (44)."""
        # pylint:disable=invalid-name
        return (
            varrho
            * math.sqrt(4 / 3)
            * torch.cos(1 / 3 * torch.acos(-q * math.sqrt(27)))
        )

    @staticmethod
    def _compute_omega1(r_min, r_opt, effective_shift):
        """Compute omega resulting from the first branch of Equation (40)."""
        return effective_shift * r_min.abs() / r_opt.abs()

    @staticmethod
    def _compute_omega2(r_min, r_opt, effective_shift):
        """Compute omega resulting from the second branch of Equation (40)."""
        omega_hat_hat = (2 - r_opt.abs()) / (2 - r_min.abs())
        ratio_ = (effective_shift.abs() - r_opt.abs()) / (2 - r_opt.abs())
        ratio_to_omega_hat_hat = ratio_ ** omega_hat_hat
        omega_hat = r_min.abs() + (2 - r_min.abs()) * ratio_to_omega_hat_hat
        omega2 = effective_shift.sign() * omega_hat
        return omega2

    @override(dynamics)
    def state_transition(self, domain, phi_idx, system_response, effective_shift):
        old_domain = domain

        # (0) compute new domain
        domain = self._compute_domain(old_domain, effective_shift)
        # (1) if domain change: system_response <- advantageous
        system_response = torch.where(
            domain != old_domain, torch.ones_like(system_response), system_response
        )
        # (2) compute & apply turn direction
        phi_idx = phi_idx + self._compute_angular_step(
            domain, phi_idx, system_response, effective_shift
        )
        # (3) Update system response if necessary
        system_response = self._updated_system_response(phi_idx, system_response)
        # (4) apply symmetry
        phi_idx = self._apply_symmetry(phi_idx)
        # (5) if self._phi_idx == 0: reset internal state
        cond = (phi_idx == 0) & (effective_shift.abs() <= self._safe_zone)
        domain = torch.where(cond, torch.ones_like(domain), domain)
        phi_idx = torch.where(cond, torch.zeros_like(phi_idx), phi_idx)
        system_response = torch.where(
            cond, torch.ones_like(system_response), system_response
        )

        return domain, phi_idx, system_response

    @override(dynamics)
    def _compute_domain(self, domain, effective_shift):
        return torch.where(
            effective_shift.abs() <= self._safe_zone, domain, effective_shift.sign()
        )

    @override(dynamics)
    def _compute_angular_step(self, domain, phi_idx, system_response, effective_shift):
        return torch.where(
            effective_shift.abs() <= self._safe_zone,
            # cool down when effective_shift is in the safe zone
            -phi_idx.sign(),
            torch.where(
                phi_idx == -domain * self._strongest_penality_abs_idx,
                # If phi reaches the left or right limit for positive or negative domain
                # respectively, remain constant
                torch.zeros_like(phi_idx),
                # If phi is in the middle, move according to system response and domain
                system_response * effective_shift.sign(),
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
