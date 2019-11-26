"""
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

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
# pylint: disable=invalid-name
from collections import OrderedDict

import numpy as np
from gym.utils import seeding

from .goldstone.environment import environment as GoldstoneEnvironment
from .effective_action import EffectiveAction


class IDS:
    """
    Lightweight python implementation of the industrial benchmark
    Uses the same standard settings as in src/main/ressources/simTest.properties
    of java implementation
    """

    # pylint: disable=too-many-instance-attributes,missing-docstring,protected-access

    def __init__(self, p=50, stationary_p=True):
        """
        p sets the setpoint hyperparameter (between 1-100) which will
        affect the dynamics and stochasticity.

        stationary_p = False will make the setpoint vary over time. This
        will make the system more non-stationary.
        """
        self._init_p = p
        self.stationary_p = stationary_p

        self.set_seed()

        # constants
        self.maxRequiredStep = np.sin(15.0 / 180.0 * np.pi)
        self.gsBound = 1.5  # for use in equation (8) to update the effective shift
        self.gsSetPointDependency = 0.02
        self.gsScale = (
            2.0 * self.gsBound + 100.0 * self.gsSetPointDependency
        )  # scaling factor for shift

        # cost/reward weighting constants
        self.CRF = 3.0
        self.CRC = 1.0
        self.CRGS = 25.0

        self.gsEnvironment = None
        self.state = None
        self.init = False
        self._p_steps = None
        self._p_step = None
        self._p_ch = None
        self.reset()

    def set_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.gsEnvironment = GoldstoneEnvironment(
            24, self.maxRequiredStep, self.maxRequiredStep / 2.0
        )

        self.state = OrderedDict()

        self.state["o"] = np.zeros(10)  # operational cost buffer
        # Current operational cost
        # Should be updated according to equation (6) of the paper
        self.state["coc"] = 0
        self.state["fb"] = 0.0  # basic fatigue: without bifurcation aspects
        self.state["oc"] = 0  # current operational cost conv
        self.state["hg"] = 0.0  # hidden gain
        self.state["hv"] = 0.0  # hidden velocity
        self.state["he"] = 0.0  # hidden/ effective shift

        # Goldstone variables
        # Miscalibration domain. Denoted delta in the paper and starts at +1.
        self.state["gs_domain"] = self.gsEnvironment._dynamics.Domain.positive.value
        # Miscalibration System Response. Denoted psi in the paper and starts at +1.
        self.state[
            "gs_sys_response"
        ] = self.gsEnvironment._dynamics.System_Response.advantageous.value
        # Miscalibration direction. Denoted phi
        self.state["gs_phi_idx"] = 0
        self.state["ge"] = 0.0  # effective action gain beta
        self.state["ve"] = 0.0  # effective action velocity alpha
        self.state["MC"] = 0.0  # Miscalibration

        # observables
        # Table I of the Appendix mentions only setpoint, velocity, gain, shift,
        # consumption, and fatigue as the observable variables
        # self.observable_keys = ["p", "v", "g", "h", "f", "c"]
        self.observable_keys = ["p", "v", "g", "h", "f", "c", "cost", "reward"]
        self.state["p"] = self._init_p  # SetPoint
        self.state["v"] = 50.0  # Velocity
        self.state["g"] = 50.0  # Gain
        self.state["h"] = 50.0  # Shift
        self.state["f"] = 0.0  # fatigue
        self.state["c"] = 0.0  # consumption
        self.state["cost"] = 0.0  # signal/ total
        self.state["reward"] = 0.0  # reward

        self.init = True
        self.defineNewSequence()
        self.step(np.zeros(3))

    def visibleState(self):
        return np.concatenate(
            [np.array(self.state[k]).ravel() for k in self.observable_keys]
        )

    def markovState(self):
        return np.concatenate(
            [np.array(self.state[k]).ravel() for k in self.state.keys()]
        )

    def step(self, delta):
        self.updateSetPoint()
        self.addAction(delta)
        self.updateFatigue()
        self.updateCurrentOperationalCost()
        self.updateOperationalCostConvolution()
        self.updateGS()
        self.updateOperationalCosts()
        self.updateCost()

    def updateSetPoint(self):
        if self.stationary_p:
            return

        if self._p_step == self._p_steps:
            self.defineNewSequence()

        new_p = self.state["p"] + self._p_ch
        if new_p > 100 or new_p < 0:

            if self.np_random.rand() > 0.5:
                self._p_ch *= -1

        new_p = np.clip(new_p, 0, 100)

        self.state["p"] = new_p
        self._p_step += 1

    def addAction(self, delta):
        """Apply the 3-dimensional actions.

        Clips the action to [-1, 1]^3. Updates the velocity, gain and shift
        according to equations (2), (3), and (4) of the paper. The gain
        coefficient is simply dh = 20 sin(15deg) / 0.9 ≈ 5.75.

        Furthermore, the effective shift is updated using the current setpoint
        and the updated shift according to equation (8) of the paper.
        """
        # Action bounds OK
        delta = np.clip(delta, -1, 1)
        # Update velocity
        self.state["v"] = np.clip(self.state["v"] + delta[0], 0.0, 100.0)
        # Update gain
        self.state["g"] = np.clip(self.state["g"] + 10 * delta[1], 0.0, 100.0)
        # Update shift
        self.state["h"] = np.clip(
            self.state["h"]
            + ((self.maxRequiredStep / 0.9) * 100.0 / self.gsScale) * delta[2],
            0.0,
            100.0,
        )
        # Update effective shift through equation (8)
        # The scaling factor for the shift is effectively 1 / 20
        # The scaling factor for the setpoint is effectively 1 / 50
        self.state["he"] = np.clip(
            self.gsScale * self.state["h"] / 100.0
            - self.gsSetPointDependency * self.state["p"]
            - self.gsBound,
            -self.gsBound,
            self.gsBound,
        )

    def updateFatigue(self):  # pylint: disable=too-many-locals
        expLambda = 0.1
        actionTolerance = 0.05
        fatigueAmplification = 1.1
        fatigueAmplificationMax = 5.0
        fatigueAmplificationStart = 1.2

        velocity = self.state["v"]
        gain = self.state["g"]
        setpoint = self.state["p"]

        hidden_gain = self.state["hg"]
        hidden_velocity = self.state["hv"]

        effAct = EffectiveAction(velocity, gain, setpoint)
        effAct_velocity = effAct.getEffectiveVelocity()
        effAct_gain = effAct.getEffectiveGain()

        self.state["ge"] = effAct_gain
        self.state["ve"] = effAct_velocity

        noise_e_g = self.np_random.exponential(expLambda)
        noise_e_v = self.np_random.exponential(expLambda)
        noise_u_g = self.np_random.rand()
        noise_u_v = self.np_random.rand()

        noise_b_g = np.float(
            self.np_random.binomial(1, np.clip(effAct_gain, 0.001, 0.999))
        )
        noise_b_v = np.float(
            self.np_random.binomial(1, np.clip(effAct_velocity, 0.001, 0.999))
        )

        noise_gain = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_g)) - 0.5)
        noise_velocity = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_v)) - 0.5)

        noise_gain += (1 - noise_gain) * noise_u_g * noise_b_g * effAct_gain
        noise_velocity += (1 - noise_velocity) * noise_u_v * noise_b_v * effAct_velocity

        if effAct_gain <= actionTolerance:
            hidden_gain = effAct_gain
        elif hidden_gain >= fatigueAmplificationStart:
            hidden_gain = np.minimum(
                fatigueAmplificationMax, fatigueAmplification * hidden_gain
            )
        else:
            hidden_gain = 0.9 * hidden_gain + noise_gain / 3.0

        if effAct_velocity <= actionTolerance:
            hidden_velocity = effAct_velocity
        elif hidden_velocity >= fatigueAmplificationStart:
            hidden_velocity = np.minimum(
                fatigueAmplificationMax, fatigueAmplification * hidden_velocity
            )
        else:
            hidden_velocity = 0.9 * hidden_velocity + noise_velocity / 3.0

        if np.maximum(hidden_velocity, hidden_gain) == fatigueAmplificationMax:
            alpha = 1.0 / (1.0 + np.exp(-self.np_random.normal(2.4, 0.4)))
        else:
            alpha = np.maximum(noise_velocity, noise_gain)

        fb = np.maximum(0, ((30000.0 / ((5 * velocity) + 100)) - 0.01 * (gain ** 2)))
        self.state["hv"] = hidden_velocity
        self.state["hg"] = hidden_gain
        self.state["f"] = (fb * (1 + 2 * alpha)) / 3.0
        self.state["fb"] = fb

    def updateCurrentOperationalCost(self):
        """Calculate the current operational cost through equation (6) of the paper."""
        # Coefficients for calculation of equation (6)
        CostSetPoint = 2.0
        CostVelocity = 4.0
        CostGain = 2.5

        gain = self.state["g"]
        velocity = self.state["v"]
        setpoint = self.state["p"]

        # Calculation of equation (6)
        costs = CostSetPoint * setpoint + CostGain * gain + CostVelocity * velocity
        o = np.exp(costs / 100.0)
        self.state["coc"] = o

        # Update history of operational costs for future use in equation (7)
        if self.init:
            self.state["o"] += o
            self.init = False
        else:
            self.state["o"][:-1] = self.state["o"][1:]
            self.state["o"][-1] = o

    def updateOperationalCostConvolution(self):
        """Calculate the convoluted cost according to equation (7) of the paper."""
        ConvArray = np.array(
            [0.11111, 0.22222, 0.33333, 0.22222, 0.11111, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.state["oc"] = np.dot(self.state["o"], ConvArray)

    def updateGS(self):
        """Calculate the domain, response, direction, and MisCalibration.

        Computes equations (9), (10), (11), and (12) of the paper under the
        'Dynamics of mis-calibration' section.
        """
        effective_shift = self.state["he"]

        domain = self.state["gs_domain"]
        phi_idx = self.state["gs_phi_idx"]
        system_response = self.state["gs_sys_response"]

        reward, domain, phi_idx, system_response = self.gsEnvironment.state_transition(
            self.gsEnvironment._dynamics.Domain(domain),
            phi_idx,
            self.gsEnvironment._dynamics.System_Response(system_response),
            effective_shift,
        )
        self.state["MC"] = -reward
        self.state["gs_domain"] = domain.value
        self.state["gs_sys_response"] = system_response.value
        self.state["gs_phi_idx"] = phi_idx

    def updateOperationalCosts(self):
        """Calculate the consumption according to equations (19) and (20)."""
        rGS = self.state["MC"]
        # This seems to correspond to equation (19),
        # although the minus sign is mysterious.
        eNewHidden = self.state["oc"] - (self.CRGS * (rGS - 1.0))
        # This corresponds to equation (20), although the constant 0.005 is
        # different from the 0.02 written in the paper. This might result in
        # very low observational noise
        operationalcosts = eNewHidden - self.np_random.randn() * (
            1 + 0.005 * eNewHidden
        )
        self.state["c"] = operationalcosts

    def updateCost(self):
        """Calculate cost according to equation (5) of the paper."""
        fatigue = self.state["f"]
        consumption = self.state["c"]
        cost = self.CRF * fatigue + self.CRC * consumption

        self.state["cost"] = cost
        self.state["reward"] = -cost

    def defineNewSequence(self):
        """Sample new setpoint sequence parameters.

        See section III.D of the original paper for details.
        """
        length = self.np_random.randint(1, 100)
        self._p_steps = length
        self._p_step = 0
        p_ch = 2 * self.np_random.rand() - 1
        if self.np_random.rand() < 0.1:
            p_ch *= 0.0
        self._p_ch = p_ch
