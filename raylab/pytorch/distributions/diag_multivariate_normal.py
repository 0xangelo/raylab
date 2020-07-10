# pylint:disable=missing-docstring
# pylint: enable=missing-docstring
import torch
import torch.distributions as dists


class DiagMultivariateNormal(dists.Independent):
    """
    Creates a multivariate Normal (also called Gaussian) distribution
    parameterized by a mean vector and a diagonal covariance matrix.
    """

    # pylint:disable=abstract-method

    def __init__(self, loc, scale_diag, validate_args=None):
        base_distribution = dists.Normal(loc=loc, scale=scale_diag)
        super().__init__(
            base_distribution, reinterpreted_batch_ndims=1, validate_args=validate_args
        )

    def reproduce(self, event):
        """Produce a reparametrized sample with the same value as `event`."""
        loc = self.base_dist.loc
        scale_diag = self.base_dist.scale
        with torch.no_grad():
            eps = (event - loc) / scale_diag
        return loc + scale_diag * eps
