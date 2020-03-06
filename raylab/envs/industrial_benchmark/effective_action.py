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


class EffectiveAction:
    # pylint:disable=missing-docstring

    def __init__(self, velocity, gain, setpoint):
        self.setpoint = setpoint
        self.effective_velocity = self.calc_effective_velocity(velocity, gain, setpoint)
        self.effective_gain = self.calc_effective_gain(gain, setpoint)

    def calc_effective_velocity(self, velocity, gain, setpoint):
        min_alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(100, setpoint), self.calceffective_b(0, setpoint)
        )
        max_alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(0, setpoint), self.calceffective_b(100, setpoint)
        )
        alpha_unscaled = self.calc_effective_velocity_unscaled(
            self.calceffective_a(velocity, setpoint),
            self.calceffective_b(gain, setpoint),
        )
        return (alpha_unscaled - min_alpha_unscaled) / (
            max_alpha_unscaled - min_alpha_unscaled
        )

    def calc_effective_gain(self, gain, setpoint):
        min_beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(100, setpoint)
        )
        max_beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(0, setpoint)
        )
        beta_unscaled = self.calc_effective_gain_unscaled(
            self.calceffective_b(gain, setpoint)
        )
        return (beta_unscaled - min_beta_unscaled) / (
            max_beta_unscaled - min_beta_unscaled
        )

    @staticmethod
    def calceffective_a(velocity, setpoint):
        return velocity + 101.0 - setpoint

    @staticmethod
    def calceffective_b(gain, setpoint):
        return gain + 1.0 + setpoint

    @staticmethod
    def calc_effective_velocity_unscaled(effective_a, effective_b):
        return (effective_b + 1.0) / effective_a

    @staticmethod
    def calc_effective_gain_unscaled(effective_b):
        return 1.0 / effective_b

    def get_effective_velocity(self):
        return self.effective_velocity

    def get_effective_gain(self):
        return self.effective_gain
