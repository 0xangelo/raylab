"""
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Alexander Hentschel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np

from .nlgp import NLGP


class RewardFunction:
    """
    generates the reward function for fixed phi
    works ONLY WITH SCALAR inputs
    Input:
     * phi: angle in Radians
     * max_required_step: the maximum required step by the optimal policy;
        must be positive
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, phi, max_required_step):
        self.phi = phi
        self.max_required_step = max_required_step
        if max_required_step <= 0:
            raise ValueError("Value for argument max_required_step must be positive")

        self._reward_function = self._reward_function_factory(phi, max_required_step)
        self._vectorized_reward_function = np.vectorize(self._reward_function)
        self.optimum_radius = self._compute_optimal_radius(phi, max_required_step)
        self.optimum_value = self._reward_function(self.optimum_radius)

    def reward(self, radius):
        """Compute the reward for a given effective shift."""
        return self._reward_function(radius)

    @staticmethod
    def _rad_transformation_factory(opt_rad, min_rad):
        """Compute Equation (40) for given optimal and minimum radii."""

        def tsf(effective_shift):
            # This also ensures that we don't get an imaginary number when
            # exponentiating (abs(effective_shift) - abs(opt_rad))
            if abs(effective_shift) <= abs(opt_rad):
                return effective_shift * abs(min_rad) / abs(opt_rad)

            omega_hat_hat = (2 - abs(opt_rad)) / (2 - abs(min_rad))  # Equation (42)
            scaling = (2 - abs(min_rad)) / (2 - abs(opt_rad)) ** omega_hat_hat
            # Equation (41)
            omega_hat = (
                abs(min_rad)
                + scaling * (abs(effective_shift) - abs(opt_rad)) ** omega_hat_hat
            )
            return np.sign(effective_shift) * omega_hat

        return tsf

    @staticmethod
    def _compute_optimal_radius(phi, max_required_step):
        """Compute Equation (43)."""
        phi = np.mod(phi, 2 * np.pi)

        # max_required_step = 2 * safe_zone
        opt = max(abs(np.sin(phi)), max_required_step)
        if phi >= np.pi:
            opt *= -1
        return opt

    def _reward_function_factory(self, phi, max_required_step):
        """
        generates the reward function for fixed phi
        works ONLY WITH SCALAR inputs
        Input:
         * phi: angle in Radians
         * max_required_step: the maximum required step by the optimal policy;
            must be positive
        """
        nlgp = NLGP()
        # use 2-pi-symmetry to move phi in domain [0,2pi]
        phi = np.mod(phi, 2 * np.pi)
        # the desired radius at which we want the global optimim to be:
        opt_rad = self._compute_optimal_radius(phi, max_required_step)
        #
        # the radius of minimum in NLGP:
        min_rad = nlgp.global_minimum_radius(phi)
        #
        # radius transformation such that minimum is moved to desired value
        transform = self._rad_transformation_factory(opt_rad, min_rad)
        transform = np.vectorize(transform)
        return lambda r: nlgp.polar_nlgp(transform(r), phi)
