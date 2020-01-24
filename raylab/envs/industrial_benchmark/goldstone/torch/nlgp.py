"""PyTorch implementation of NLGP."""
import torch
import numpy as np


class TorchNLGP:
    """Differentiable version of Goldstone's nlgp."""

    def __init__(self):
        u_0 = np.cbrt(1 + np.sqrt(2)) / np.sqrt(3)
        r_0 = u_0 + 1 / (3 * u_0)

        lmbd = 2 * r_0 ** 2 - r_0 ** 4 + 8 * np.sqrt(2 / 27.0) * r_0

        self._norm_alpha = 2 / lmbd
        self._norm_beta = 1 / lmbd
        self._norm_kappa = -8 * np.sqrt(2 / 27.0) / lmbd
        self._phi_b = np.pi / 4.0
        self._qh_b = -np.sqrt(1 / 27.0)

    def polar_nlgp(self, radius, phi):
        """Apply Equation (17)
        Function value of normalized, linearly biased Goldstone Potential
        in polar coordinates:
          * radius in R
          * angle in Radians
        """
        rsq = radius ** 2
        return (
            -self._norm_alpha * rsq
            + self._norm_beta * (rsq ** 2)
            + self._norm_kappa * phi.sin() * radius
        )

    def global_minimum_radius(self, phi):
        """
        returns the radius r_0 along phi-axis where NLG has minimal function value, i.e.
            r_0 = argmin_{r} polar_nlgp(r,phi)
        angle phi in Radians
        """
        q_h = self._norm_kappa * phi.sin().abs() / (8 * self._norm_beta)

        signum_phi = phi.sin().sign()

        signum_phi = torch.where(
            signum_phi == 0, torch.ones_like(signum_phi), signum_phi
        )

        aux = (-signum_phi * q_h + torch.sqrt(q_h * q_h - 1 / 27)) ** (1 / 3)
        r_0 = torch.where(
            q_h <= self._qh_b,
            aux + 1 / (3 * aux),
            signum_phi
            * np.sqrt(4 / 3)
            * torch.cos(1 / 3 * torch.acos(-q_h * np.sqrt(27))),
        )
        return r_0
