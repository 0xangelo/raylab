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
from collections import OrderedDict

import numpy as np
from gym.utils import seeding

from .goldstone.environment import GoldstoneEnvironment
from .effective_action import EffectiveAction


class IDS:
    """
    Lightweight python implementation of the industrial benchmark
    Uses the same standard settings as in src/main/ressources/simTest.properties
    of java implementation
    """

    # pylint: disable=too-many-instance-attributes,missing-docstring,protected-access
    # cost/reward weighting constants
    CRF = 3.0
    CRC = 1.0
    CRGS = 25.0

    # fatigue dynamics constants
    exp_lambda = 0.1
    action_tolerance = 0.05
    fatigue_amplification = 1.1
    fatigue_amplification_max = 5.0
    fatigue_amplification_start = 1.2

    # Coefficients for calculation of equation (6)
    cost_setpoint = 2.0
    cost_velocity = 4.0
    cost_gain = 2.5

    def __init__(self, p=50, stationary_p=True, miscalibration=True):
        """
        p sets the setpoint hyperparameter (between 1-100) which will
        affect the dynamics and stochasticity.

        stationary_p = False will make the setpoint vary over time. This
        will make the system more non-stationary.
        """
        self._init_p = p
        self.stationary_p = stationary_p
        self._miscalibration = miscalibration

        self.set_seed()

        # constants
        self.max_required_step = np.sin(15.0 / 180.0 * np.pi)
        self.gs_bound = 1.5  # for use in equation (8) to update the effective shift
        self.gs_setpoint_dependency = 0.02
        self.gs_scale = (
            2.0 * self.gs_bound + 100.0 * self.gs_setpoint_dependency
        )  # scaling factor for shift

        self.gs_env = None
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
        self.gs_env = GoldstoneEnvironment(
            24, self.max_required_step, self.max_required_step / 2.0  # safe zone
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
        self.state["gs_domain"] = self.gs_env._dynamics.Domain.positive.value
        # Miscalibration System Response. Denoted psi in the paper and starts at +1.
        self.state[
            "gs_sys_response"
        ] = self.gs_env._dynamics.SystemResponse.advantageous.value
        # Miscalibration direction. Denoted phi
        self.state["gs_phi_idx"] = 0
        self.state["ge"] = 0.0  # effective action gain beta
        self.state["ve"] = 0.0  # effective action velocity alpha
        self.state["MC"] = 0.0  # Miscalibration

        # observables
        # Table I of the Appendix mentions only setpoint, velocity, gain, shift,
        # fatigue, and comsuption as the observable variables
        self.observable_keys = ["p", "v", "g", "h", "f", "c"]
        self.state["p"] = self._init_p  # SetPoint
        self.state["v"] = 50.0  # Velocity
        self.state["g"] = 50.0  # Gain
        self.state["h"] = 50.0  # Shift
        self.state["f"] = 0.0  # fatigue
        self.state["c"] = 0.0  # consumption
        self.state["cost"] = 0.0  # signal/ total
        self.state["reward"] = 0.0  # reward

        self.init = True
        self.define_new_sequence()
        self.step(np.zeros(3))

    def visible_state(self):
        """Return the observable state as in Table I of the Appendix."""
        return np.concatenate(
            [np.array(self.state[k]).ravel() for k in self.observable_keys]
        )

    def full_state(self):
        """Return all current state variables."""
        hid_vars = [k for k in self.state.keys() if k not in set(self.observable_keys)]
        return np.concatenate(
            [self.visible_state()] + [np.array(self.state[k]).ravel() for k in hid_vars]
        )

    def minimal_markov_state(self):
        """Return the minimal Markovian state as in Table I of the Appendix."""
        obs_vars = [np.array(self.state[k]).ravel() for k in self.observable_keys]
        op_costs = [self.state["o"][1:]]
        miscalibration = [
            np.array(self.state[k]).ravel()
            for k in ["gs_domain", "gs_sys_response", "gs_phi_idx"]
        ]
        fatigue = [np.array(self.state[k]).ravel() for k in ["hv", "hg"]]
        return np.concatenate(obs_vars + op_costs + miscalibration + fatigue)

    def step(self, delta):
        self.update_setpoint()
        self.add_action(delta)
        self.update_fatigue()
        self.update_current_operational_cost()
        self.update_operational_cost_convolution()
        self.update_gs()
        self.update_operational_costs()
        self.update_cost()

    def update_setpoint(self):
        if self.stationary_p:
            return

        if self._p_step == self._p_steps:
            self.define_new_sequence()

        new_p = self.state["p"] + self._p_ch
        if new_p > 100 or new_p < 0:

            if self.np_random.rand() > 0.5:
                self._p_ch *= -1

        new_p = np.clip(new_p, 0, 100)

        self.state["p"] = new_p
        self._p_step += 1

    def add_action(self, delta):
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
            + ((self.max_required_step / 0.9) * 100.0 / self.gs_scale) * delta[2],
            0.0,
            100.0,
        )
        if self._miscalibration:
            # Update effective shift through equation (8)
            # The scaling factor for the shift is effectively 1 / 20
            # The scaling factor for the setpoint is effectively 1 / 50
            self.state["he"] = np.clip(
                self.gs_scale * self.state["h"] / 100.0
                - self.gs_setpoint_dependency * self.state["p"]
                - self.gs_bound,
                -self.gs_bound,
                self.gs_bound,
            )
        else:
            self.state["he"] = np.sin(np.pi * self.state["gs_phi_idx"] / 12)

    def update_fatigue(self):  # pylint: disable=too-many-locals
        velocity = self.state["v"]
        gain = self.state["g"]
        setpoint = self.state["p"]

        hidden_gain = self.state["hg"]
        hidden_velocity = self.state["hv"]

        eff_act = EffectiveAction(velocity, gain, setpoint)
        eff_act_velocity = eff_act.get_effective_velocity()
        eff_act_gain = eff_act.get_effective_gain()

        self.state["ge"] = eff_act_gain
        self.state["ve"] = eff_act_velocity

        noise_e_g = self.np_random.exponential(self.exp_lambda)
        noise_e_v = self.np_random.exponential(self.exp_lambda)
        noise_e_g = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_g)) - 0.5)
        noise_e_v = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_v)) - 0.5)

        noise_u_g = self.np_random.rand()
        noise_u_v = self.np_random.rand()

        noise_b_g = np.float(
            self.np_random.binomial(1, np.clip(eff_act_gain, 0.001, 0.999))
        )
        noise_b_v = np.float(
            self.np_random.binomial(1, np.clip(eff_act_velocity, 0.001, 0.999))
        )

        noise_gain = noise_e_g + (1 - noise_e_g) * noise_u_g * noise_b_g * eff_act_gain
        noise_velocity = (
            noise_e_v + (1 - noise_e_v) * noise_u_v * noise_b_v * eff_act_velocity
        )

        if eff_act_gain <= self.action_tolerance:
            hidden_gain = eff_act_gain
        elif hidden_gain >= self.fatigue_amplification_start:
            hidden_gain = np.minimum(
                self.fatigue_amplification_max, self.fatigue_amplification * hidden_gain
            )
        else:
            hidden_gain = 0.9 * hidden_gain + noise_gain / 3.0

        if eff_act_velocity <= self.action_tolerance:
            hidden_velocity = eff_act_velocity
        elif hidden_velocity >= self.fatigue_amplification_start:
            hidden_velocity = np.minimum(
                self.fatigue_amplification_max,
                self.fatigue_amplification * hidden_velocity,
            )
        else:
            hidden_velocity = 0.9 * hidden_velocity + noise_velocity / 3.0

        if np.maximum(hidden_velocity, hidden_gain) >= self.fatigue_amplification_max:
            alpha = 1.0 / (1.0 + np.exp(-self.np_random.normal(2.4, 0.4)))
        else:
            alpha = np.maximum(noise_velocity, noise_gain)

        basic_fatigue = np.maximum(
            0, ((30000.0 / ((5 * velocity) + 100)) - 0.01 * (gain ** 2))
        )
        self.state["hv"] = hidden_velocity
        self.state["hg"] = hidden_gain
        self.state["f"] = (basic_fatigue * (1 + 2 * alpha)) / 3.0
        self.state["fb"] = basic_fatigue

    def update_current_operational_cost(self):
        """Calculate the current operational cost through equation (6) of the paper."""
        gain = self.state["g"]
        velocity = self.state["v"]
        setpoint = self.state["p"]

        # Calculation of equation (6)
        costs = (
            self.cost_setpoint * setpoint
            + self.cost_gain * gain
            + self.cost_velocity * velocity
        )
        coc = np.exp(costs / 100.0)
        self.state["coc"] = coc

        # Update history of operational costs for future use in equation (7)
        if self.init:
            self.state["o"] += coc
            self.init = False
        else:
            self.state["o"][1:] = self.state["o"][:-1]
            self.state["o"][0] = coc

    def update_operational_cost_convolution(self):
        """Calculate the convoluted cost according to equation (7) of the paper."""
        conv_array = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.11111, 0.22222, 0.33333, 0.22222, 0.11111]
        )
        self.state["oc"] = np.dot(self.state["o"], conv_array)

    def update_gs(self):
        """Calculate the domain, response, direction, and MisCalibration.

        Computes equations (9), (10), (11), and (12) of the paper under the
        'Dynamics of mis-calibration' section.
        """
        effective_shift = self.state["he"]

        domain = self.state["gs_domain"]
        phi_idx = self.state["gs_phi_idx"]
        system_response = self.state["gs_sys_response"]

        reward, domain, phi_idx, system_response = self.gs_env.state_transition(
            self.gs_env._dynamics.Domain(domain),
            phi_idx,
            self.gs_env._dynamics.SystemResponse(system_response),
            effective_shift,
        )
        self.state["MC"] = -reward
        self.state["gs_domain"] = domain.value
        self.state["gs_sys_response"] = system_response.value
        self.state["gs_phi_idx"] = phi_idx

    def update_operational_costs(self):
        """Calculate the consumption according to equations (19) and (20)."""
        rgs = self.state["MC"]
        # This seems to correspond to equation (19),
        # although the minus sign is mysterious.
        # hidden_cost = self.state["oc"] - (self.CRGS * (rgs - 1.0))
        hidden_cost = self.state["oc"] + self.CRGS * rgs
        # This corresponds to equation (20), although the constant 0.005 is
        # different from the 0.02 written in the paper. This might result in
        # very low observational noise
        # operationalcosts = hidden_cost - self.np_random.randn() * (
        #     1 + 0.005 * hidden_cost
        # )
        operationalcosts = hidden_cost + self.np_random.randn() * (
            1 + 0.02 * hidden_cost
        )
        self.state["c"] = operationalcosts

    def update_cost(self):
        """Calculate cost according to equation (5) of the paper."""
        fatigue = self.state["f"]
        consumption = self.state["c"]
        cost = self.CRF * fatigue + self.CRC * consumption

        self.state["cost"] = cost
        self.state["reward"] = -cost

    def define_new_sequence(self):
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
