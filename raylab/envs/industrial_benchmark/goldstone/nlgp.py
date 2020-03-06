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
from scipy.special import cbrt


class nlgp:
    def __init__(self):
        u0 = cbrt(1 + np.sqrt(2)) / np.sqrt(3)  # Equation (34)
        r0 = u0 + 1 / (3 * u0)  # Equation (35)

        lmbd = 2 * r0 ** 2 - r0 ** 4 + 8 * np.sqrt(2 / 27.0) * r0  # Equation (36)

        self.__norm_alpha = 2 / lmbd  # Equation (37)
        self.__norm_beta = 1 / lmbd  # Equation (38)
        self.__norm_kappa = -8 * np.sqrt(2 / 27.0) / lmbd  # Equation (39)
        self.__phi_b = np.pi / 4.0  # Unused
        self.__qh_b = -np.sqrt(1 / 27.0)  # Threshold for equation (44)

    def polar_nlgp(self, omega, phi):
        """Apply Equation (17)
        Function value of normalized, linearly biased Goldstone Potential
        in polar coordinates:
          * omega is radius in R
          * angle in Radians
        """
        rsq = np.square(omega)
        return (
            -self.__norm_alpha * rsq
            + self.__norm_beta * np.square(rsq)
            + self.__norm_kappa * np.sin(phi) * omega
        )

    def global_minimum_radius(self, phi):
        """
        returns the radius r0 along phi-axis where NLG has minimal function value, i.e.
            r0 = argmin_{r} polar_nlgp(r,phi)
        angle phi in Radians
        """
        f = np.vectorize(self.__global_minimum_radius)
        return f(phi)

    def __global_minimum_radius(self, phi):
        # Equation (46)
        qh = self.__norm_kappa * abs(np.sin(phi)) / (8 * self.__norm_beta)
        # Equation (47)
        signum_phi = np.sign(np.sin(phi))

        if signum_phi == 0:
            signum_phi = 1

        if qh <= self.__qh_b:
            u = cbrt(-signum_phi * qh + np.sqrt(qh * qh - 1 / 27))  # Equation (45)
            r0 = u + 1 / (3 * u)  # First case of Equation (44)
        else:
            # Second case of Equation (44)
            r0 = (
                signum_phi
                * np.sqrt(4 / 3)
                * np.cos(1 / 3 * np.arccos(-qh * np.sqrt(27)))
            )
        return r0
