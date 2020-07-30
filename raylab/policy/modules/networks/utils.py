# pylint:disable=missing-module-docstring
import torch
import torch.nn as nn
from torch import Tensor


class TensorStandardScaler(nn.Module):
    # pylint:disable=line-too-long
    """Shifts and scales 1D inputs for model networks.

    Based on MBPO's `TensorStandardScaler`_

    .. _`TensorStandardScaler`: https://github.com/JannerM/mbpo/blob/28b1e3b1382dcda34b421961286b59f77770d48c/mbpo/models/utils.py#L14

    Args:
        input_size: Size of the inputs to the scaler.

    Notes:
        Loss functions for model ensembles already expand the inputs for the
        ensemble size, so the input size used here should correspond to the raw
        observation/action size.
    """
    # pylint:enable=line-too-long

    def __init__(self, input_size: int):
        super().__init__()
        self.register_buffer("scaler_mu", torch.zeros(input_size))
        self.register_buffer("scaler_sigma", torch.ones(input_size))
        self.fitted = False

    @torch.jit.export
    def fit(self, data: Tensor):
        """Assigns the data's mean and stddev to internal buffers.

        Runs two ops, one for assigning the mean of the data to the internal
        mean, and another for assigning the standard deviation of the data to
        the internal standard deviation.

        Arguments:
            data: Tensor containing the input
        """
        # Reduce all but the last dimension
        # pylint:disable=unnecessary-comprehension
        reduction_dims = [i for i in range(data.dim() - 1)]
        # pylint:enable=unnecessary-comprehension

        data_mu = torch.mean(data, dim=reduction_dims, keepdim=False)
        data_sigma = torch.std(data, dim=reduction_dims, keepdim=False)
        data_sigma = torch.where(
            data_sigma < 1e-12, torch.ones_like(data_sigma), data_sigma
        )

        self.scaler_mu.copy_(data_mu)
        self.scaler_sigma.copy_(data_sigma)
        self.fitted = True

    def forward(self, data: Tensor) -> Tensor:
        """Transforms the input matrix data using the parameters of this scaler.

        Args:
            data: Tensor containing the points to be transformed.

        Returns:
            The transformed dataset.
        """
        # pylint:disable=arguments-differ
        return (data - self.scaler_mu) / self.scaler_sigma

    @torch.jit.export
    def inverse_transform(self, data: Tensor) -> Tensor:
        """Undoes the transformation performed by this scaler.

        Args:
            data: Tensor containing the points to be transformed.

        Returns:
            The transformed dataset.
        """
        return self.scaler_sigma * data + self.scaler_mu
