# pylint: disable=missing-docstring
import torch.distributions as dists


class DiagMultivariateNormal(dists.Independent):
    """
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a diagonal covariance matrix.
    """

    # pylint: disable=abstract-method

    def __init__(self, loc, scale_diag, validate_args=None):
        base_distribution = dists.Normal(loc=loc, scale=scale_diag)
        super().__init__(
            base_distribution, reinterpreted_batch_ndims=1, validate_args=validate_args
        )
