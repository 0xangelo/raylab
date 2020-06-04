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

Implementations of various coupling layers.

Slightly modified from:
https://github.com/bayesiains/nsf/blob/master/nde/transforms/coupling.py
"""
import warnings
from abc import ABCMeta
from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
from ray.rllib.utils import override

from .abstract import ConditionalTransform
from .nonlinearities import PiecewiseRationalQuadraticCDF
from .splines import DEFAULT_MIN_BIN_HEIGHT
from .splines import DEFAULT_MIN_BIN_WIDTH
from .splines import DEFAULT_MIN_DERIVATIVE
from .splines import unconstrained_rational_quadratic_spline
from .utils import sum_rightmost


class CouplingTransform(ConditionalTransform, metaclass=ABCMeta):
    """A base class for coupling layers. Supports 1D inputs (*, D), where D >= 2."""

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        *,
        event_dim=1,
    ):
        """
        Constructor.
        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 1:
            raise ValueError("Coupling dimension must greater than 1.")

        super().__init__(event_dim=event_dim)

        self.features = len(mask)
        indexes = torch.arange(self.features)
        self.register_buffer("identity_features", indexes.masked_select(mask <= 0))
        self.register_buffer("transform_features", indexes.masked_select(mask > 0))
        # Number of features that condition the transform.
        self.num_identity_features = len(self.identity_features)
        # Number of features that are conditionally transformed.
        self.num_transform_features = len(self.transform_features)
        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self.transform_dim_multiplier,
        )
        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @override(ConditionalTransform)
    def encode(self, inputs, params: Dict[str, torch.Tensor]):
        identity_split = inputs.index_select(-1, self.identity_features)
        transform_split = inputs.index_select(-1, self.transform_features)

        transform_params = self.transform_net(identity_split, params)
        transform_split, logabsdet = self.coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split, params
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs.index_copy_(-1, self.identity_features, identity_split)
        outputs.index_copy_(-1, self.transform_features, transform_split)

        return outputs, logabsdet

    @override(ConditionalTransform)
    def decode(self, inputs, params: Dict[str, torch.Tensor]):
        identity_split = inputs.index_select(-1, self.identity_features)
        transform_split = inputs.index_select(-1, self.transform_features)

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split, params
            )

        transform_params = self.transform_net(identity_split, params)
        transform_split, logabsdet_split = self.coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs.index_copy_(-1, self.identity_features, identity_split)
        outputs.index_copy_(-1, self.transform_features, transform_split)

        return outputs, logabsdet

    @property
    @abstractmethod
    def transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""

    @abstractmethod
    def coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""

    @abstractmethod
    def coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""


class AffineCouplingTransform(CouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    @property
    @override(CouplingTransform)
    def transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[..., self.num_transform_features :]
        shift = transform_params[..., : self.num_transform_features]

        # Obtain Scale
        # Option 1: use exp()
        log_scale = unconstrained_scale * 0.01
        scale = log_scale.exp()

        # Option 2: use softplus()
        # scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
        # log_scale = scale.log()

        # Option 3: use sigmoid()
        # scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        # log_scale = scale.log()

        return scale, shift, log_scale

    @override(CouplingTransform)
    def coupling_transform_forward(self, inputs, transform_params):
        scale, shift, log_scale = self._scale_and_shift(transform_params)
        outputs = inputs * scale + shift
        logabsdet = sum_rightmost(log_scale, self.event_dim)
        return outputs, logabsdet

    @override(CouplingTransform)
    def coupling_transform_inverse(self, inputs, transform_params):
        scale, shift, log_scale = self._scale_and_shift(transform_params)
        outputs = (inputs - shift) / scale
        logabsdet = -sum_rightmost(log_scale, self.event_dim)
        return outputs, logabsdet


class AdditiveCouplingTransform(AffineCouplingTransform):
    """An additive coupling layer, i.e. an affine coupling layer without scaling.
    Reference:
    > L. Dinh et al., NICE:  Non-linear  Independent  Components  Estimation,
    > arXiv:1410.8516, 2014.
    """

    @property
    @override(CouplingTransform)
    def transform_dim_multiplier(self):
        return 1

    @override(AffineCouplingTransform)
    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.ones_like(shift)
        return scale, shift, torch.zeros_like(scale)


class PiecewiseCouplingTransform(CouplingTransform):
    """
    Base class for piecewise coupling transforms, which define their inverse externally.
    """

    @override(CouplingTransform)
    def coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, reverse=False)

    @override(CouplingTransform)
    def coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, reverse=True)

    def _coupling_transform(self, inputs, transform_params, reverse: bool = False):

        # For batched 1D data, reshape transform_params from (*, D*?) to (*, D, ?)
        transform_params = torch.reshape(transform_params, inputs.shape + (-1,))

        outputs, logabsdet = self._piecewise_cdf(
            inputs, transform_params, reverse=reverse
        )

        return outputs, sum_rightmost(logabsdet, self.event_dim)

    @abstractmethod
    def _piecewise_cdf(self, inputs, transform_params, reverse: bool = False):
        pass


class PiecewiseRQSCouplingTransform(PiecewiseCouplingTransform):
    """Piecewise Rational Quadratic Spline Coupling Transform."""

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        **kwargs,
    ):
        # pylint:disable=too-many-arguments
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tail_bound = tail_bound

        if apply_unconditional_transform:

            def unconditional_transform(features):
                return PiecewiseRationalQuadraticCDF(
                    shape=[features] + (img_shape if img_shape else []),
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                    min_derivative=min_derivative,
                )

        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
            **kwargs,
        )

        if hasattr(self.transform_net, "hidden_features"):
            self.scale = 1.0 / np.sqrt(self.transform_net.hidden_features)
        else:
            self.scale = 1.0
            warnings.warn(
                "Inputs to the softmax are not scaled down: init might be bad."
            )

    @property
    @override(CouplingTransform)
    def transform_dim_multiplier(self):
        # Always use linear tails
        return self.num_bins * 3 - 1

    def _piecewise_cdf(self, inputs, transform_params, reverse: bool = False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        unnormalized_widths *= self.scale
        unnormalized_heights *= self.scale

        # Always use linear tails
        return unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
