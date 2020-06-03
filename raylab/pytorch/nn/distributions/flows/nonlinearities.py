"""
MIT License

Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios

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

Implementations of invertible non-linearities.

Slightly modified from:
https://github.com/bayesiains/nsf/blob/master/nde/transforms/nonlinearities.py
"""
# pylint:disable=missing-class-docstring
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils import override

from .abstract import CompositeTransform
from .abstract import InverseTransform
from .abstract import Transform
from .splines import DEFAULT_MIN_BIN_HEIGHT
from .splines import DEFAULT_MIN_BIN_WIDTH
from .splines import DEFAULT_MIN_DERIVATIVE
from .splines import unconstrained_rational_quadratic_spline
from .utils import sum_rightmost


class Tanh(Transform):
    def encode(self, inputs):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = sum_rightmost(logabsdet, self.event_dim)
        return outputs, logabsdet

    def decode(self, inputs):
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = sum_rightmost(logabsdet, self.event_dim)
        return outputs, logabsdet


class LogTanh(Transform):
    """
    Tanh with unbounded output. Constructed by selecting a cut_point, and replacing
    values to the right of cut_point with alpha * log(beta * x), and to the left of
    -cut_point with -alpha * log(-beta * x). alpha and beta are set to match the value
    and the first derivative of tanh at cut_point.
    """

    def __init__(self, cut_point=1):
        if cut_point <= 0:
            raise ValueError("Cut point must be positive.")
        super().__init__()

        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)

        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp(
            (np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha
        )

    def encode(self, inputs):
        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = sum_rightmost(logabsdet, self.event_dim)

        return outputs, logabsdet

    def decode(self, inputs):

        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log(
            (1 + inputs[mask_middle]) / (1 - inputs[mask_middle])
        )
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = (
            -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        )
        logabsdet[mask_left] = (
            -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        )
        logabsdet = sum_rightmost(logabsdet, self.event_dim)

        return outputs, logabsdet


class LeakyReLU(Transform):
    def __init__(self, negative_slope=1e-2):
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        super().__init__()
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    def encode(self, inputs):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = self.log_negative_slope * mask
        logabsdet = sum_rightmost(logabsdet, self.event_dim)
        return outputs, logabsdet

    def decode(self, inputs):
        outputs = F.leaky_relu(inputs, negative_slope=(1 / self.negative_slope))
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = sum_rightmost(logabsdet, self.event_dim)
        return outputs, logabsdet


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.temperature = torch.Tensor([temperature])

    def encode(self, inputs):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = sum_rightmost(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs),
            self.event_dim,
        )
        return outputs, logabsdet

    def decode(self, inputs):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise ValueError("Input outside domain")

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -sum_rightmost(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs),
            self.event_dim,
        )
        return outputs, logabsdet


class Logit(InverseTransform):
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))


class CauchyCDF(Transform):
    def encode(self, inputs):
        outputs = (1 / np.pi) * torch.atan(inputs) + 0.5
        logabsdet = sum_rightmost(
            -np.log(np.pi) - torch.log(1 + inputs ** 2), self.event_dim
        )
        return outputs, logabsdet

    def decode(self, inputs):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise ValueError("Input outside domain")

        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -sum_rightmost(
            -np.log(np.pi) - torch.log(1 + outputs ** 2), self.event_dim
        )
        return outputs, logabsdet


class CauchyCDFInverse(InverseTransform):
    def __init__(self):
        super().__init__(CauchyCDF())


class CompositeCDFTransform(CompositeTransform):
    def __init__(self, squashing_transform, cdf_transform):
        super().__init__(
            [squashing_transform, cdf_transform, InverseTransform(squashing_transform)]
        )


class PiecewiseRationalQuadraticCDF(Transform):
    # pylint:disable=too-many-instance-attributes
    def __init__(
        self,
        shape,
        num_bins=10,
        tail_bound=1.0,
        identity_init=False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        event_dim=1,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(event_dim=event_dim)

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))
            constant = np.log(np.exp(1 - min_derivative) - 1)
            self.unnormalized_derivatives = nn.Parameter(
                torch.empty(*shape, num_bins - 1).fill_(constant)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_bins - 1)
            )

    def _spline(self, inputs, inverse: bool = False):
        batch_shape = inputs.shape[: -self.event_dim]
        unnormalized_widths = self.unnormalized_widths.expand(
            batch_shape + self.unnormalized_widths.shape
        )
        unnormalized_heights = self.unnormalized_heights.expand(
            batch_shape + self.unnormalized_heights.shape
        )
        unnormalized_derivatives = self.unnormalized_derivatives.expand(
            batch_shape + self.unnormalized_derivatives.shape
        )

        # Always use linear tails
        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        return outputs, sum_rightmost(logabsdet, self.event_dim)

    def encode(self, inputs):
        return self._spline(inputs, inverse=False)

    def decode(self, inputs):
        return self._spline(inputs, inverse=True)


##################################################################
# Custom
##################################################################


class TanhTransform(Transform):
    """Transform via the mapping :math:`y = \frac{e^x - e^{-x}} {e^x + e^{-x}}`."""

    # pylint:disable=arguments-out-of-order

    @override(Transform)
    def encode(self, inputs):
        outputs = torch.tanh(inputs)
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        # torch.finfo(torch.float32).tiny
        to_log1 = torch.clamp(1 + inputs, min=1.1754943508222875e-38)
        to_log2 = torch.clamp(1 - inputs, min=1.1754943508222875e-38)
        outputs = (torch.log(to_log1) - torch.log(to_log2)) / 2
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        # Taken from spinningup's implementation of SAC
        return sum_rightmost(
            2 * (math.log(2) - inputs - F.softplus(-2 * inputs)), self.event_dim
        )


class SigmoidTransform(Transform):
    # pylint:disable=missing-docstring,arguments-out-of-order

    @override(Transform)
    def encode(self, inputs):
        outputs = inputs.sigmoid()
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        to_log = inputs.clamp(min=1.1754943508222875e-38)
        outputs = to_log.log() - (-to_log).log1p()
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        return sum_rightmost(-F.softplus(-inputs) - F.softplus(inputs), self.event_dim)


class AffineTransform(Transform):
    # pylint:disable=missing-docstring,arguments-out-of-order

    def __init__(self, loc, scale, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @override(Transform)
    def encode(self, inputs):
        outputs = inputs * self.scale + self.loc
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        outputs = (inputs - self.loc) / self.scale
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        _, scale = torch.broadcast_tensors(inputs, self.scale)
        return sum_rightmost(scale.abs().log(), self.event_dim)


class TanhSquashTransform(Transform):
    """Squashes samples to the desired range using Tanh."""

    def __init__(self, low, high, event_dim=0):
        divide = AffineTransform(loc=torch.zeros_like(low), scale=2 / (high - low))
        squash = TanhTransform()
        shift = AffineTransform(loc=(high + low) / 2, scale=(high - low) / 2)
        compose = CompositeTransform([divide, squash, shift], event_dim=event_dim)
        super().__init__(cond_transform=compose)


class SigmoidSquashTransform(Transform):
    """Squashes samples to the desired range using Sigmoid."""

    def __init__(self, low, high, event_dim=0):
        divide = AffineTransform(loc=torch.zeros_like(low), scale=1 / (high - low))
        squash = SigmoidTransform()
        shift = AffineTransform(loc=low, scale=(high - low))
        compose = CompositeTransform([divide, squash, shift], event_dim=event_dim)
        super().__init__(cond_transform=compose)
